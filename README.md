# Worker Segmentation Cluster (BrainNav)

Repository ini berisi data-plane worker untuk distributed brain segmentation.

## Topologi Worker

- PC-A: 2x RTX 5080, subscribe mode homogen + heterogen + fallback legacy
- PC-B: 1x RTX 5080, subscribe mode homogen + fallback legacy
- PC-C: 1x RTX 2080 (template/standby), subscribe mode heterogen + fallback legacy

Master services berada di node master:

- RabbitMQ: 100.79.202.62:5674
- Redis: 100.79.202.62:6381

## Struktur Folder

```text
worker-segmentation/
  PC-A/
  PC-B/
  PC-C/
```

Setiap folder PC berisi stack worker mandiri:

- worker.py
- config.py
- chunk_processor.py
- model_loader.py
- Dockerfile
- docker-compose.yml
- .env

## Mekanisme Queue

Queue yang digunakan:

- segmentation_tasks (legacy)
- segmentation_tasks_homogen
- segmentation_tasks_heterogen

Worker membaca CLUSTER_MODES dari env/compose untuk subscribe queue mode, dan selalu subscribe queue legacy untuk backward compatibility.

## Instrumentasi Runtime

Setiap chunk menghasilkan log runtime ke Redis key:

- segmentation:chunklog:{task_id}:{chunk_id}

Isi log mencakup:

- worker_name
- gpu_type
- cluster_mode
- queue_wait_ms
- inference_ms
- total_chunk_ms

Prometheus metrics tersedia di endpoint /metrics (port 8000 per container).

## Menjalankan Worker

Jalankan dari folder node yang diinginkan.

Contoh PC-A:

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-A"
docker compose up -d --build
```

Contoh PC-B:

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-B"
docker compose up -d --build
```

Contoh PC-C (saat hardware sudah siap):

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-C"
docker compose up -d --build
```

## Catatan Penting

- Jangan ubah prefetch_count=1 di worker.py (justifikasi pull-based scheduling).
- Jangan ubah pipeline UNETR inference (scope TA hanya deployment/distributed inference).
- Pastikan best_metric_model.pth tersedia pada tiap node sebelum start container.
- Jangan commit kredensial atau file model besar ke git.
