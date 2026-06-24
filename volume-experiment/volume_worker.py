#!/usr/bin/env python3
"""
Volume-Level Distributed Segmentation Worker (standalone, tanpa BrainNav)
========================================================================

Worker untuk eksperimen distributed system IEEE. Berbeda dari worker chunk
lama (worker-segmentation/PC-A), worker ini memproses SATU VOLUME 3D UTUH per
task -- sesuai strategi volume-level pada paper. Logika inferensi mengikuti
`worker-segmentation/scripts_nnunet/test_volume_inference.py` yang sudah
terbukti menghasilkan mask valid (chunk 2D mentah -> mask kosong/Dice 0).

ALUR
----
1. Connect RabbitMQ (consume queue volume) + Redis (tulis hasil & log).
2. Load nnU-Net 2D ensemble predictor SEKALI (warm) -> tetap di memori.
3. Per message:
   - decode payload volume (base64 .nii.gz  ATAU  referensi file lokal)
   - jalankan inferensi volume penuh (5-fold ensemble)
   - ukur queue_wait_ms (transfer+antri) dan inference_ms (komputasi)
   - tulis ringkasan hasil + log timing ke Redis (dikelompokkan per run_id)
   - increment counter selesai -> ACK
4. Expose Prometheus /metrics + /health (Flask) pada METRICS_PORT.

prefetch_count=1 => fair dispatch (pull-based): worker cepat (RTX 5080)
otomatis mengambil lebih banyak task daripada worker lambat (RTX 3050Ti).

ENV (lihat .env.example)
------------------------
  RABBITMQ_HOST/PORT/USER/PASS/VHOST, VOLUME_QUEUE
  REDIS_URL
  NNUNET_MODEL_FOLDER, NNUNET_FOLDS, DEVICE
  WORKER_NAME, GPU_TYPE, METRICS_PORT
  PAYLOAD_MODE = base64 | reference
  SCAN_DIR  (untuk PAYLOAD_MODE=reference: folder berisi <scan_id>.nii.gz)
"""
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, Tuple

import numpy as np

try:
    import nibabel as nib
    import pika
    import redis
    import torch
    from flask import Flask, jsonify
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as e:  # pragma: no cover
    print(f"[ERROR] dependency belum lengkap: {e}")
    print("        Jalankan di dalam venv nnU-Net GPU (mis. nnunet_gpu_env).")
    sys.exit(1)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("volume-worker")


# --------------------------------------------------------------------------
# Konfigurasi dari environment
# --------------------------------------------------------------------------
def _folds_from_env(raw: str) -> Tuple[int, ...]:
    vals = [int(x) for x in raw.split(",") if x.strip().isdigit()]
    return tuple(vals) if vals else (0, 1, 2, 3, 4)


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "100.110.113.24")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5674"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "brainnav")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "BrainNav_Secure_2025!")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "brainnav_vhost")
VOLUME_QUEUE = os.getenv("VOLUME_QUEUE", "segmentation_volume_tasks")

REDIS_URL = os.getenv("REDIS_URL", "redis://:BrainNav_Secure_2025!@100.110.113.24:6381/0")
RESULT_TTL = int(os.getenv("RESULT_TTL", "86400"))  # 24 jam untuk post-hoc

NNUNET_MODEL_FOLDER = os.getenv("NNUNET_MODEL_FOLDER", "./nnUnet-Papavero")
NNUNET_FOLDS = _folds_from_env(os.getenv("NNUNET_FOLDS", "0,1,2,3,4"))
CHECKPOINT = os.getenv("NNUNET_CHECKPOINT", "checkpoint_final.pth")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

WORKER_NAME = os.getenv("WORKER_NAME", f"worker-{os.getpid()}")
GPU_TYPE = os.getenv("GPU_TYPE", "UNKNOWN")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

PAYLOAD_MODE = os.getenv("PAYLOAD_MODE", "reference").strip().lower()
SCAN_DIR = os.getenv("SCAN_DIR", "./scans")
# Jika di-set, worker menyimpan SATU mask hasil prediksi per run_id (file
# pertama yang ia proses untuk run itu) sebagai NIfTI. Berguna untuk
# membandingkan kualitas segmentasi antar node/konfigurasi pada slice yang sama.
SAVE_MASK_DIR = os.getenv("SAVE_MASK_DIR", "").strip()


# --------------------------------------------------------------------------
# Prometheus metrics
# --------------------------------------------------------------------------
LABELS = ["worker_name", "gpu_type"]
M_TASKS = Counter("volseg_tasks_processed_total", "Volume tasks processed", LABELS)
M_FAILED = Counter("volseg_tasks_failed_total", "Volume tasks failed", LABELS)
M_INFER = Histogram(
    "volseg_inference_seconds", "Inference time per volume (compute only)", LABELS,
    buckets=(1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300),
)
M_QWAIT = Histogram(
    "volseg_queue_wait_seconds", "Queue wait + transfer time per task", LABELS,
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
M_TOTAL = Histogram(
    "volseg_total_seconds", "Total handling time per task", LABELS,
    buckets=(1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600),
)
M_ACTIVE = Gauge("volseg_active_tasks", "Currently active tasks", LABELS)


# --------------------------------------------------------------------------
# Inferensi volume (mengikuti test_volume_inference.py)
# --------------------------------------------------------------------------
def to_nnunet_input(vol: np.ndarray):
    """(any-order volume) -> (1, Z, Y, X) + axis_z (axis paling kecil = slice)."""
    axis_z = int(np.argmin(vol.shape))
    vol_zyx = np.moveaxis(vol, axis_z, 0)
    arr = vol_zyx[np.newaxis].astype(np.float32)
    return arr, axis_z


def run_volume_inference(predictor, vol: np.ndarray, spacing):
    """Return (mask_zyx uint8, axis_z). mask dalam orientasi (Z,Y,X)."""
    arr, axis_z = to_nnunet_input(vol)
    sp = list(spacing)
    sp_z = sp[axis_z]
    sp_rest = [sp[i] for i in range(len(sp)) if i != axis_z]
    spacing_zyx = [sp_z] + sp_rest if len(sp_rest) == 2 else [999.0, 1.0, 1.0]
    mask = predictor.predict_single_npy_array(
        arr, {"spacing": spacing_zyx},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    return np.asarray(mask).astype(np.uint8), axis_z


def load_volume_from_task(task: Dict[str, Any]) -> Tuple[np.ndarray, list, np.ndarray]:
    """Ambil (volume, spacing, affine) dari task sesuai PAYLOAD_MODE."""
    if PAYLOAD_MODE == "reference":
        scan_id = task["scan_id"]
        path = os.path.join(SCAN_DIR, f"{scan_id}.nii.gz")
        nii = nib.load(path)
        return nii.get_fdata().astype(np.float32), list(nii.header.get_zooms()[:3]), nii.affine

    # default: base64 dari isi file .nii.gz di dalam message
    raw = base64.b64decode(task["volume_b64"])
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        nii = nib.load(tmp_path)
        vol = nii.get_fdata().astype(np.float32)
        spacing = list(nii.header.get_zooms()[:3])
        affine = nii.affine
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return vol, spacing, affine


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
class VolumeWorker:
    def __init__(self):
        self.labels = {"worker_name": WORKER_NAME, "gpu_type": GPU_TYPE}
        self.redis = redis.Redis.from_url(REDIS_URL)
        self.redis.ping()
        logger.info("Redis OK: %s", REDIS_URL.split("@")[-1])

        logger.info("Loading nnU-Net predictor (folds=%s, device=%s)...", NNUNET_FOLDS, DEVICE)
        t0 = time.perf_counter()
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            device=torch.device(DEVICE), verbose=False, allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            NNUNET_MODEL_FOLDER, use_folds=NNUNET_FOLDS, checkpoint_name=CHECKPOINT,
        )
        logger.info("Predictor loaded in %.1fs (warm).", time.perf_counter() - t0)

        self._saved_runs = set()  # run_id yang sudah disimpan mask-nya oleh worker ini
        if SAVE_MASK_DIR:
            os.makedirs(SAVE_MASK_DIR, exist_ok=True)
            logger.info("SAVE_MASK_DIR aktif: %s (1 mask/run/worker)", SAVE_MASK_DIR)

        self._setup_rabbitmq()

    def _setup_rabbitmq(self):
        creds = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST, port=RABBITMQ_PORT, virtual_host=RABBITMQ_VHOST,
            credentials=creds, heartbeat=900, blocked_connection_timeout=600,
        )
        self.conn = pika.BlockingConnection(params)
        self.ch = self.conn.channel()
        self.ch.queue_declare(queue=VOLUME_QUEUE, durable=True,
                              arguments={"x-message-ttl": 3600000})
        self.ch.basic_qos(prefetch_count=1)  # fair dispatch (pull-based)
        logger.info("RabbitMQ OK: %s:%s queue=%s", RABBITMQ_HOST, RABBITMQ_PORT, VOLUME_QUEUE)

    def _handle(self, ch, method, properties, body):
        recv_ns = time.time_ns()
        try:
            task = json.loads(body)
        except json.JSONDecodeError:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        task_id = task.get("task_id", "?")
        run_id = task.get("run_id", "default")
        published_ns = int(task.get("published_at_ns", recv_ns))
        queue_wait_ms = max(0, int((recv_ns - published_ns) / 1e6))

        M_ACTIVE.labels(**self.labels).inc()
        total_t0 = time.perf_counter()
        try:
            vol, spacing, affine = load_volume_from_task(task)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            inf_t0 = time.perf_counter()
            mask, axis_z = run_volume_inference(self.predictor, vol, spacing)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            inference_ms = int((time.perf_counter() - inf_t0) * 1000)

            fg = int((mask > 0).sum())
            total_ms = int((time.perf_counter() - total_t0) * 1000)
            done_ns = time.time_ns()

            # Simpan 1 mask per run (orientasi & affine asli) untuk perbandingan gambar.
            if SAVE_MASK_DIR and run_id not in self._saved_runs:
                try:
                    mask_orig = np.moveaxis(mask, 0, axis_z)
                    fname = f"{run_id}__{WORKER_NAME}__{task.get('scan_id','scan')}.nii.gz"
                    nib.save(nib.Nifti1Image(mask_orig.astype(np.uint8), affine),
                             os.path.join(SAVE_MASK_DIR, fname))
                    self._saved_runs.add(run_id)
                    logger.info("Mask disimpan: %s (fg=%d)", fname, fg)
                except Exception as se:
                    logger.warning("Gagal simpan mask: %s", se)

            log = {
                "task_id": task_id, "run_id": run_id,
                "scan_id": task.get("scan_id", ""),
                "idx": task.get("idx", -1), "total": task.get("total", -1),
                "worker_name": WORKER_NAME, "gpu_type": GPU_TYPE,
                "queue_wait_ms": queue_wait_ms, "inference_ms": inference_ms,
                "total_ms": total_ms, "fg_voxels": fg,
                "volume_shape": list(vol.shape),
                "published_at_ns": published_ns, "recv_ns": recv_ns, "done_ns": done_ns,
            }
            self.redis.setex(f"segmentation:exp:log:{run_id}:{task_id}",
                             RESULT_TTL, json.dumps(log))
            self.redis.incr(f"segmentation:exp:done:{run_id}")
            self.redis.expire(f"segmentation:exp:done:{run_id}", RESULT_TTL)

            M_QWAIT.labels(**self.labels).observe(queue_wait_ms / 1000.0)
            M_INFER.labels(**self.labels).observe(inference_ms / 1000.0)
            M_TOTAL.labels(**self.labels).observe(total_ms / 1000.0)
            M_TASKS.labels(**self.labels).inc()

            logger.info("[%s] task=%s idx=%s inf=%dms qwait=%dms fg=%d",
                        WORKER_NAME, task_id, task.get("idx"), inference_ms, queue_wait_ms, fg)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error("FAIL task=%s: %s", task_id, e)
            logger.error(traceback.format_exc())
            M_FAILED.labels(**self.labels).inc()
            retry = int(task.get("retry_count", 0)) < 3
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=retry)
        finally:
            M_ACTIVE.labels(**self.labels).dec()

    def start(self):
        self.ch.basic_consume(queue=VOLUME_QUEUE, on_message_callback=self._handle, auto_ack=False)
        logger.info("Worker %s (%s) menunggu task di '%s'...", WORKER_NAME, GPU_TYPE, VOLUME_QUEUE)
        try:
            self.ch.start_consuming()
        except KeyboardInterrupt:
            self.ch.stop_consuming()
        finally:
            self.conn.close()


# --------------------------------------------------------------------------
# Flask: /health + /metrics
# --------------------------------------------------------------------------
app = Flask(__name__)
_worker_ready = {"ok": False}


@app.route("/health")
def health():
    return jsonify({"status": "healthy" if _worker_ready["ok"] else "initializing",
                    "worker": WORKER_NAME, "gpu": GPU_TYPE, "device": DEVICE}), \
        (200 if _worker_ready["ok"] else 503)


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


def main():
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=METRICS_PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    logger.info("Metrics/health di port %d", METRICS_PORT)

    worker = VolumeWorker()
    _worker_ready["ok"] = True
    worker.start()


if __name__ == "__main__":
    main()
