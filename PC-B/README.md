# Segmentation Worker

Python worker service untuk brain segmentation menggunakan model UNETR. Worker ini consume tasks dari RabbitMQ, proses segmentasi, dan simpan hasil ke Redis.

## Architecture

```
RabbitMQ Queue → Worker → UNETR Model → Redis Cache
```

### Processing Pipeline

1. **Consume Task**: Receive chunk segmentation task dari RabbitMQ
2. **Preprocess**: Convert 2D chunk (256×256) → 3D volume (128×128×128)
   - Resize dengan bicubic interpolation
   - Depth replication
   - Gaussian depth variation untuk simulasi adjacent slices
3. **Inference**: Run UNETR model
   - Input: (1, 1, 128, 128, 128)
   - Output: (1, 2, 128, 128, 128) - 2 channels: background + brain
4. **Postprocess**: Convert 3D output → 2D segmentation mask (256×256)
   - Extract center slice
   - Apply sigmoid + threshold (0.5)
   - Resize ke original chunk size
5. **Store Result**: Write ke Redis dengan TTL
6. **Update Progress**: Increment counter
7. **ACK Task**: Acknowledge ke RabbitMQ

## Files

```
segmentation-worker/
├── worker.py           # Main worker process
├── model_loader.py     # UNETR model loader
├── chunk_processor.py  # 2D→3D→2D processing pipeline
├── config.py           # Configuration from environment variables
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
└── README.md          # This file
```

## Configuration

Environment variables (set in `.env` or `docker-compose.yml`):

```bash
# Worker Identity
WORKER_NAME=worker-1
CUDA_VISIBLE_DEVICES=0

# RabbitMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=brainnav
RABBITMQ_PASS=brainnav_secure_password_change_me
RABBITMQ_VHOST=brainnav_vhost
RABBITMQ_QUEUE=segmentation_tasks

# Redis
REDIS_URL=redis://:password@redis:6379/0
SEGMENTATION_RESULT_TTL=3600  # Result cache TTL in seconds

# Model
MODEL_PATH=/models/best_metric_model.pth
DEVICE=cuda  # or cpu

# Processing
PATCH_SIZE=128
OVERLAP_RATIO=0.25
BATCH_SIZE=1
NUM_WORKERS=2

# Logging
LOG_LEVEL=INFO
```

## Running Locally

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ensure model file exists
ls best_metric_model.pth
```

### Run Worker

```bash
# Set environment variables
export WORKER_NAME=worker-local
export MODEL_PATH=../best_metric_model.pth
export RABBITMQ_HOST=localhost
export REDIS_URL=redis://localhost:6379/0

# Run
python worker.py
```

### Health Check

```bash
# Health status
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Running in Docker

### Build Image

```bash
docker build -t brainnav-worker .
```

### Run Container

```bash
docker run -d \
  --name worker-1 \
  --gpus device=0 \
  -e WORKER_NAME=worker-1 \
  -e RABBITMQ_HOST=rabbitmq \
  -e REDIS_URL=redis://redis:6379/0 \
  -v $(pwd)/../best_metric_model.pth:/models/best_metric_model.pth:ro \
  -p 8001:8000 \
  brainnav-worker
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `segmentation_tasks_processed_total` - Total tasks successfully processed
- `segmentation_tasks_failed_total` - Total failed tasks
- `segmentation_processing_seconds` - Processing time distribution (histogram)
- `segmentation_active_tasks` - Currently active tasks (gauge)
- `model_inference_seconds` - Model inference time (histogram)
- `redis_write_seconds` - Redis write time (histogram)
- `chunk_preprocess_seconds` - Preprocessing time (histogram)

### Logs

```bash
# Docker logs
docker logs worker-1 -f

# File logs
tail -f /app/logs/worker.log
```

### Health Check

```bash
# Check health
curl http://localhost:8000/health

# Expected response (healthy):
{
  "status": "healthy",
  "worker_name": "worker-1",
  "device": "cuda:0",
  "model_loaded": true,
  "timestamp": 1706123456.789
}

# Unhealthy states:
# - "initializing": Worker starting up
# - "rabbitmq_disconnected": RabbitMQ connection lost
# - "unhealthy": General error
```

## Development

### Testing Locally

```bash
# Install dev dependencies
pip install pytest pytest-cov ipython

# Run tests
pytest tests/

# Test single chunk processing
python -c "
import numpy as np
from chunk_processor import ChunkProcessor
from model_loader import ModelLoader

model_loader = ModelLoader('best_metric_model.pth', 'cuda')
processor = ChunkProcessor(model_loader.model, model_loader.device, None)

chunk = np.random.rand(256, 256).astype(np.float32)
result = processor.process_chunk(chunk)
print(f'Input: {chunk.shape}, Output: {result.shape}')
"
```

### Debugging

```bash
# Enable debug logs
export LOG_LEVEL=DEBUG

# Run worker with verbose output
python worker.py

# Inspect Redis
redis-cli
> KEYS segmentation:*
> GET segmentation:result:task123:chunk0
```

## Performance

### Expected Latency

Per chunk (256×256):
- Preprocessing: 50-100ms
- Inference (UNETR): 1.5-2.0s (RTX 5080)
- Postprocessing: 50-100ms
- Redis write: 10-20ms
- **Total: ~2-3 seconds**

### GPU Memory Usage

- Model size: ~1.2GB (UNETR weights)
- Batch processing: ~3-4GB (peak)
- Total per worker: ~4-5GB GPU memory

### CPU Usage

- 2 cores recommended
- ~40-60% utilization during inference

## Troubleshooting

### Worker Won't Start

```bash
# Check RabbitMQ connection
telnet rabbitmq 5672

# Check Redis connection
redis-cli -h redis ping

# Check GPU availability
nvidia-smi

# Verify model file
ls -lh /models/best_metric_model.pth
```

### High Latency

```bash
# Check GPU utilization
nvidia-smi dmon

# Check queue depth
rabbitmqctl list_queues -p brainnav_vhost

# Reduce batch size
export BATCH_SIZE=1

# Check CPU/Memory
htop
```

### Memory Errors

```bash
# Out of GPU memory
nvidia-smi  # Check GPU memory usage

# Solutions:
# 1. Reduce batch size
# 2. Clear GPU cache: python -c "import torch; torch.cuda.empty_cache()"
# 3. Restart worker

# Out of RAM
# Check memory usage: free -h
# Increase Docker memory limit
```

### Task Failures

```bash
# Check error logs
docker logs worker-1 | grep ERROR

# Check Redis for error details
redis-cli GET segmentation:error:taskid:chunkid

# Verify RabbitMQ DLX
rabbitmqctl list_queues -p brainnav_vhost | grep dlx
```

## API Reference

### Task Message Format (from RabbitMQ)

```json
{
  "task_id": "seg-abc123def456",
  "chunk_id": 0,
  "chunk_data": "base64_encoded_numpy_array",
  "chunk_shape": [256, 256],
  "position": {
    "x": 0,
    "y": 0,
    "width": 256,
    "height": 256,
    "overlap_x": 32,
    "overlap_y": 32
  },
  "total_chunks": 4,
  "retry_count": 0
}
```

### Redis Result Format

Key: `segmentation:result:{task_id}:{chunk_id}`
Value: Base64-encoded numpy array

Metadata Key: `segmentation:metadata:{task_id}:{chunk_id}`
Value:
```json
{
  "shape": [256, 256],
  "dtype": "uint8",
  "timestamp": 1706123456.789,
  "worker": "worker-1"
}
```

## License

Part of BrainNav project. See root LICENSE.
