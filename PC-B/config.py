"""
Configuration for Segmentation Worker
"""
import os
from dataclasses import dataclass


@dataclass
class WorkerConfig:
    """Worker configuration from environment variables"""
    
    # RabbitMQ
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_pass: str
    rabbitmq_vhost: str
    rabbitmq_queue: str
    
    # Redis
    redis_url: str
    result_ttl: int  # Time to live for results in seconds
    
    # Model
    model_path: str
    device: str  # 'cuda' or 'cpu'
    
    # Processing
    patch_size: int
    overlap_ratio: float
    batch_size: int
    num_workers: int
    
    # Logging
    log_level: str
    worker_name: str
    
    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        """Load configuration from environment variables"""
        return cls(
            # RabbitMQ
            rabbitmq_host=os.getenv('RABBITMQ_HOST', 'localhost'),
            rabbitmq_port=int(os.getenv('RABBITMQ_PORT', '5672')),
            rabbitmq_user=os.getenv('RABBITMQ_USER', 'brainnav'),
            rabbitmq_pass=os.getenv('RABBITMQ_PASS', 'brainnav_secure_password'),
            rabbitmq_vhost=os.getenv('RABBITMQ_VHOST', 'brainnav_vhost'),
            rabbitmq_queue=os.getenv('RABBITMQ_QUEUE', 'segmentation_tasks'),
            
            # Redis
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            result_ttl=int(os.getenv('SEGMENTATION_RESULT_TTL', '3600')),
            
            # Model
            model_path=os.getenv('MODEL_PATH', '/models/best_metric_model.pth'),
            device=os.getenv('DEVICE', 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'),
            
            # Processing
            patch_size=int(os.getenv('PATCH_SIZE', '128')),
            overlap_ratio=float(os.getenv('OVERLAP_RATIO', '0.25')),
            batch_size=int(os.getenv('BATCH_SIZE', '4')),
            num_workers=int(os.getenv('NUM_WORKERS', '4')),
            
            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            worker_name=os.getenv('WORKER_NAME', f'worker-{os.getpid()}'),
        )
