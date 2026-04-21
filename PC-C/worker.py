#!/usr/bin/env python3
"""
BrainNav Segmentation Worker
Consumes tasks from RabbitMQ, performs brain segmentation using UNETR model,
and stores results in Redis.

Architecture:
1. Connect to RabbitMQ (message queue for task distribution)
2. Connect to Redis (for storing segmentation results)
3. Load UNETR model into memory
4. Consume tasks from queue:
   - Preprocess 2D chunk to 3D volume with Gaussian depth variation
   - Run UNETR inference
   - Postprocess to 2D segmentation mask
   - Store result in Redis
   - Update progress counter
   - ACK task

Metrics published to Prometheus on port 8000/metrics
Health check available on port 8000/health
"""

import os
import sys
import json
import time
import logging
import traceback
import threading
from typing import Dict, Any, List

import numpy as np
import pika
import redis
from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from tenacity import retry, stop_after_attempt, wait_exponential

from model_loader import ModelLoader
from chunk_processor import ChunkProcessor
from config import WorkerConfig

# Create logs directory
os.makedirs('/app/logs', exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/worker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('segmentation-worker')

# Prometheus Metrics
METRIC_LABELS = ['worker_name', 'gpu_type', 'cluster_mode']

TASKS_PROCESSED = Counter(
    'segmentation_tasks_processed_total',
    'Total segmentation tasks processed',
    METRIC_LABELS,
)
TASKS_FAILED = Counter(
    'segmentation_tasks_failed_total',
    'Total segmentation tasks failed',
    METRIC_LABELS,
)
PROCESSING_TIME = Histogram(
    'segmentation_processing_seconds',
    'Time spent processing segmentation',
    METRIC_LABELS,
)
ACTIVE_TASKS = Gauge(
    'segmentation_active_tasks',
    'Number of currently active segmentation tasks',
    METRIC_LABELS,
)
MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Time spent on model inference',
    METRIC_LABELS,
)
REDIS_WRITE_TIME = Histogram(
    'redis_write_seconds',
    'Time spent writing to Redis',
    METRIC_LABELS,
)
CHUNK_PREPROCESS_TIME = Histogram(
    'chunk_preprocess_seconds',
    'Time spent preprocessing chunks',
    METRIC_LABELS,
)

# Flask app for health check and metrics
app = Flask(__name__)

# Global variables for health check
_worker = None
_processor = None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        if _worker is None or _processor is None:
            return jsonify({'status': 'initializing'}), 503
        
        # Check Redis connection
        _worker.redis_client.ping()
        
        # Check RabbitMQ connection
        if _worker.rabbitmq_channel is None or _worker.rabbitmq_channel.is_closed:
            return jsonify({'status': 'rabbitmq_disconnected'}), 503
        
        return jsonify({
            'status': 'healthy',
            'worker_name': _worker.worker_name,
            'device': str(_processor.device),
            'model_loaded': True,
            'timestamp': time.time()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        return jsonify({'error': str(e)}), 500


class SegmentationWorker:
    """Main worker class for brain segmentation"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_name = config.worker_name
        self.gpu_type = config.gpu_type
        self.subscribed_queues = self._build_subscribed_queues()
        
        # Initialize Redis
        self.redis_client = redis.Redis.from_url(config.redis_url)
        logger.info(f"Connected to Redis: {config.redis_url}")
        
        # Initialize RabbitMQ
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self._setup_rabbitmq()
        
        # Initialize model and processor
        logger.info(f"Loading model from {config.model_path}")
        self.model_loader = ModelLoader(config.model_path, config.device)
        self.processor = ChunkProcessor(
            self.model_loader.model,
            self.model_loader.device,
            config
        )
        
        logger.info(f"Worker {self.worker_name} initialized successfully")
        logger.info(f"GPU type: {self.gpu_type}")
        logger.info(f"Device: {self.model_loader.device}")
        logger.info(f"Subscribed queues: {', '.join(self.subscribed_queues)}")
        logger.info(f"Model loaded: {config.model_path}")

    def _build_subscribed_queues(self) -> List[str]:
        queue_map = {
            'homogen': self.config.queue_homogen,
            'heterogen': self.config.queue_heterogen,
            'legacy': self.config.queue_legacy,
        }

        queues: List[str] = []
        for mode in self.config.cluster_modes:
            mapped = queue_map.get(mode.lower().strip())
            if mapped and mapped not in queues:
                queues.append(mapped)

        # Backward compatibility: always listen legacy queue.
        for fallback in [self.config.queue_legacy, self.config.rabbitmq_queue]:
            if fallback and fallback not in queues:
                queues.append(fallback)

        return queues

    def _metric_labels(self, cluster_mode: str) -> Dict[str, str]:
        return {
            'worker_name': self.worker_name,
            'gpu_type': self.gpu_type,
            'cluster_mode': cluster_mode,
        }

    def _resolve_cluster_mode(self, task_data: Dict[str, Any]) -> str:
        raw_mode = str(task_data.get('cluster_mode', '')).strip().lower()
        if raw_mode in ('homogen', 'heterogen', 'legacy'):
            return raw_mode
        return 'legacy'

    def _compute_queue_wait_ms(self, published_at: Any) -> int:
        published_at_ns = _safe_int(published_at, 0)
        if published_at_ns <= 0:
            return 0

        now_ns = time.time_ns()
        wait_ns = max(0, now_ns - published_at_ns)
        return int(wait_ns / 1_000_000)
    
    def _setup_rabbitmq(self):
        """Setup RabbitMQ connection and channel"""
        try:
            credentials = pika.PlainCredentials(
                self.config.rabbitmq_user,
                self.config.rabbitmq_pass
            )
            parameters = pika.ConnectionParameters(
                host=self.config.rabbitmq_host,
                port=self.config.rabbitmq_port,
                virtual_host=self.config.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.rabbitmq_connection = pika.BlockingConnection(parameters)
            self.rabbitmq_channel = self.rabbitmq_connection.channel()
            
            # Declare queues with settings
            queue_args = {
                'x-message-ttl': 1800000,  # 30 minutes
                'x-max-priority': 10,
            }
            for queue_name in sorted(set(self.subscribed_queues + [
                self.config.queue_homogen,
                self.config.queue_heterogen,
                self.config.queue_legacy,
            ])):
                if queue_name:
                    self.rabbitmq_channel.queue_declare(
                        queue=queue_name,
                        durable=True,
                        arguments=queue_args,
                    )
            
            # Keep fair dispatch for pull-based dynamic scheduling.
            self.rabbitmq_channel.basic_qos(prefetch_count=1)
            
            logger.info(f"Connected to RabbitMQ: {self.config.rabbitmq_host}")
            
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _store_result_in_redis(
        self,
        task_id: str,
        chunk_id: int,
        result: np.ndarray,
        position: Dict[str, Any],
        cluster_mode: str,
        queue_wait_ms: int,
        inference_ms: int,
        total_chunk_ms: int,
    ):
        """Store segmentation result in Redis with retry logic"""
        result_key = f"segmentation:result:{task_id}:{chunk_id}"
        labels = self._metric_labels(cluster_mode)
        
        # Serialize numpy array
        result_bytes = result.astype(np.float32).tobytes()
        
        with REDIS_WRITE_TIME.labels(**labels).time():
            # Store in Redis with TTL
            self.redis_client.setex(
                result_key,
                self.config.result_ttl,
                result_bytes,
            )

            # Store metadata including chunk position and instrumentation.
            metadata_key = f"segmentation:metadata:{task_id}:{chunk_id}"
            metadata = {
                'shape': result.shape,
                'dtype': str(result.dtype),
                'timestamp': time.time(),
                'worker_name': self.worker_name,
                'gpu_type': self.gpu_type,
                'cluster_mode': cluster_mode,
                'position': position,
                'queue_wait_ms': queue_wait_ms,
                'inference_ms': inference_ms,
                'total_chunk_ms': total_chunk_ms,
            }
            self.redis_client.setex(
                metadata_key,
                self.config.result_ttl,
                json.dumps(metadata),
            )

            chunk_log_key = f"segmentation:chunklog:{task_id}:{chunk_id}"
            chunk_log = {
                'task_id': task_id,
                'chunk_id': chunk_id,
                'worker_name': self.worker_name,
                'gpu_type': self.gpu_type,
                'cluster_mode': cluster_mode,
                'queue_wait_ms': queue_wait_ms,
                'inference_ms': inference_ms,
                'total_chunk_ms': total_chunk_ms,
                'timestamp': time.time(),
            }
            self.redis_client.setex(
                chunk_log_key,
                self.config.result_ttl,
                json.dumps(chunk_log),
            )
        
        logger.debug(f"Stored result for task {task_id}, chunk {chunk_id}")
    
    def _update_progress(self, task_id: str, chunk_id: int, total_chunks: int):
        """Update task progress in Redis"""
        # Increment completed chunks
        completed_key = f"segmentation:completed:{task_id}"
        completed = self.redis_client.incr(completed_key)
        self.redis_client.expire(completed_key, self.config.result_ttl)
        
        # Calculate progress percentage
        progress = int((completed / total_chunks) * 100)
        
        # Update progress
        progress_key = f"segmentation:progress:{task_id}"
        self.redis_client.setex(
            progress_key,
            self.config.result_ttl,
            str(progress)
        )
        
        logger.info(f"Task {task_id}: {completed}/{total_chunks} chunks completed ({progress}%)")
    
    def process_task(self, task_data: Dict[str, Any]) -> bool:
        """
        Process a single segmentation task
        
        Args:
            task_data: Task data containing task_id, chunk_id, chunk_data, etc.
        
        Returns:
            bool: True if successful, False otherwise
        """
        task_id = task_data.get('task_id')
        chunk_id = task_data.get('chunk_id')
        cluster_mode = self._resolve_cluster_mode(task_data)
        labels = self._metric_labels(cluster_mode)
        
        # Validate required fields
        if task_id is None or chunk_id is None:
            logger.error("Missing required fields: task_id or chunk_id")
            return False
        
        try:
            ACTIVE_TASKS.labels(**labels).inc()
            start_time = time.time()
            total_start_ns = time.time_ns()
            queue_wait_ms = self._compute_queue_wait_ms(task_data.get('published_at'))
            
            logger.info(
                "Processing task %s, chunk %s (mode=%s, queue_wait_ms=%d)",
                task_id,
                chunk_id,
                cluster_mode,
                queue_wait_ms,
            )
            
            # Extract chunk data
            chunk_data_b64 = task_data.get('chunk_data')
            if chunk_data_b64 is None:
                logger.error(f"Missing chunk_data for task {task_id}")
                return False
            
            position = task_data.get('position') or {}
            total_chunks = task_data.get('total_chunks', 1)
            
            # Decode chunk data
            import base64
            preprocess_start = time.time()
            chunk_bytes = base64.b64decode(chunk_data_b64)
            chunk_shape = tuple(task_data.get('chunk_shape', [256, 256]))
            chunk_array = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(chunk_shape)
            CHUNK_PREPROCESS_TIME.labels(**labels).observe(time.time() - preprocess_start)
            
            # Process chunk
            inference_start_ns = time.time_ns()
            segmentation_result = self.processor.process_chunk(chunk_array)
            inference_ms = int((time.time_ns() - inference_start_ns) / 1_000_000)
            MODEL_INFERENCE_TIME.labels(**labels).observe(inference_ms / 1000.0)

            total_chunk_ms = int((time.time_ns() - total_start_ns) / 1_000_000)
            
            # Store result in Redis
            self._store_result_in_redis(
                task_id=task_id,
                chunk_id=chunk_id,
                result=segmentation_result,
                position=position,
                cluster_mode=cluster_mode,
                queue_wait_ms=queue_wait_ms,
                inference_ms=inference_ms,
                total_chunk_ms=total_chunk_ms,
            )
            
            # Update progress
            self._update_progress(task_id, chunk_id, total_chunks)
            
            # Update metrics
            processing_time = time.time() - start_time
            PROCESSING_TIME.labels(**labels).observe(processing_time)
            TASKS_PROCESSED.labels(**labels).inc()
            
            logger.info(
                "Task %s, chunk %s completed in %.2fs (mode=%s, inference_ms=%d)",
                task_id,
                chunk_id,
                processing_time,
                cluster_mode,
                inference_ms,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process task {task_id}, chunk {chunk_id}: {e}")
            logger.error(traceback.format_exc())
            TASKS_FAILED.labels(**labels).inc()
            
            # Store error in Redis
            error_key = f"segmentation:error:{task_id}:{chunk_id}"
            error_data = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'worker': self.worker_name,
                'gpu_type': self.gpu_type,
                'cluster_mode': cluster_mode,
                'timestamp': time.time()
            }
            self.redis_client.setex(error_key, 3600, json.dumps(error_data))
            
            return False
            
        finally:
            ACTIVE_TASKS.labels(**labels).dec()
    
    def callback(self, ch, method, properties, body):
        """RabbitMQ message callback"""
        try:
            task_data = json.loads(body)
            
            success = self.process_task(task_data)
            
            if success:
                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Reject and requeue (with limit)
                requeue = task_data.get('retry_count', 0) < 3
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=requeue)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Remove invalid message
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            logger.error(traceback.format_exc())
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    
    def start(self):
        """Start consuming messages from RabbitMQ"""
        try:
            if self.rabbitmq_channel is None:
                raise RuntimeError("RabbitMQ channel not initialized")
            
            logger.info(
                "Worker %s starting consumers for queues: %s",
                self.worker_name,
                ', '.join(self.subscribed_queues),
            )
            
            # Keep QoS=1 as required for fair dispatch.
            self.rabbitmq_channel.basic_qos(prefetch_count=1)  # Fair dispatch
            for queue_name in self.subscribed_queues:
                self.rabbitmq_channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=self.callback,
                    auto_ack=False,
                )
                logger.info("Subscribed to queue: %s", queue_name)
            
            logger.info(f"Worker {self.worker_name} ready to receive messages...")
            self.rabbitmq_channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Error during consuming: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.close()
    
    def close(self):
        """Close connections"""
        try:
            if self.rabbitmq_channel:
                self.rabbitmq_channel.stop_consuming()
                self.rabbitmq_channel.close()
                logger.info("RabbitMQ channel closed")
        except Exception as e:
            logger.warning(f"Error closing RabbitMQ: {e}")
        
        try:
            if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
                self.rabbitmq_connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")


if __name__ == '__main__':
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = WorkerConfig.from_env()
        
        # Initialize worker
        logger.info("Initializing worker...")
        worker = SegmentationWorker(config)
        _worker = worker
        _processor = worker.processor
        
        # Start Flask server in background thread for health checks and metrics
        logger.info("Starting Flask health check server on port 8000...")
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        
        # Give Flask a moment to start
        time.sleep(2)
        
        logger.info("Flask server started")
        logger.info(f"Worker {config.worker_name} is ready!")
        logger.info(f"Health check: http://localhost:8000/health")
        logger.info(f"Metrics: http://localhost:8000/metrics")
        
        # Start consuming tasks
        worker.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
