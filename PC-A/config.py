"""
Configuration for Segmentation Worker (nnU-Net 2D)
"""
import os
from dataclasses import dataclass
from typing import List, Tuple


def _parse_cluster_modes(raw_value: str) -> List[str]:
    modes: List[str] = []
    for mode in raw_value.split(','):
        normalized = mode.strip().lower()
        if normalized:
            modes.append(normalized)
    if not modes:
        return ['homogen']
    return list(dict.fromkeys(modes))


def _parse_folds(raw_value: str) -> Tuple[int, ...]:
    folds = []
    for f in raw_value.split(','):
        f = f.strip()
        if f.isdigit():
            folds.append(int(f))
    return tuple(folds) if folds else (0, 1, 2, 3, 4)


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
    queue_homogen: str
    queue_heterogen: str
    queue_legacy: str
    cluster_modes: List[str]

    # Redis
    redis_url: str
    result_ttl: int

    # nnU-Net model
    nnunet_model_folder: str
    nnunet_folds: Tuple[int, ...]
    device: str

    # Logging
    log_level: str
    worker_name: str
    gpu_type: str

    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        return cls(
            # RabbitMQ
            rabbitmq_host=os.getenv('RABBITMQ_HOST', 'localhost'),
            rabbitmq_port=int(os.getenv('RABBITMQ_PORT', '5672')),
            rabbitmq_user=os.getenv('RABBITMQ_USER', 'brainnav'),
            rabbitmq_pass=os.getenv('RABBITMQ_PASS', 'brainnav_secure_password'),
            rabbitmq_vhost=os.getenv('RABBITMQ_VHOST', 'brainnav_vhost'),
            rabbitmq_queue=os.getenv('RABBITMQ_QUEUE', 'segmentation_tasks'),
            queue_homogen=os.getenv('QUEUE_HOMOGEN', 'segmentation_tasks_homogen'),
            queue_heterogen=os.getenv('QUEUE_HETEROGEN', 'segmentation_tasks_heterogen'),
            queue_legacy=os.getenv('QUEUE_LEGACY', 'segmentation_tasks'),
            cluster_modes=_parse_cluster_modes(os.getenv('CLUSTER_MODES', 'homogen')),

            # Redis
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            result_ttl=int(os.getenv('SEGMENTATION_RESULT_TTL', '3600')),

            # nnU-Net model
            nnunet_model_folder=os.getenv('NNUNET_MODEL_FOLDER', '/models/nnunet_2d'),
            nnunet_folds=_parse_folds(os.getenv('NNUNET_FOLDS', '0,1,2,3,4')),
            device=os.getenv('DEVICE', 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'),

            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            worker_name=os.getenv('WORKER_NAME', f'worker-{os.getpid()}'),
            gpu_type=os.getenv('GPU_TYPE', 'UNKNOWN'),
        )
