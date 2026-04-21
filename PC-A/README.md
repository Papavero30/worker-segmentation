# PC-A Worker (RTX 5080 x2)

Node ini menjalankan 2 container worker (GPU0 dan GPU1) untuk mode homogen dan heterogen.

## Karakteristik

- Node: PC-A
- GPU: 2x RTX 5080
- Queue mode: homogen + heterogen (+ legacy fallback)
- Metric ports: 8001 (gpu0), 8002 (gpu1)

## Konfigurasi

File utama:

- .env
- docker-compose.yml

Variable penting:

- CLUSTER_MODES=homogen,heterogen
- GPU_TYPE=RTX5080
- QUEUE_HOMOGEN=segmentation_tasks_homogen
- QUEUE_HETEROGEN=segmentation_tasks_heterogen
- QUEUE_LEGACY=segmentation_tasks

## Jalankan

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-A"
docker compose up -d --build
```

## Validasi Cepat

```powershell
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8001/metrics
```

## Catatan

- PC-A dipakai di kedua skenario eksperimen (homogen dan heterogen).
- Jangan ubah mapping queue di docker-compose tanpa sinkronisasi ke master.
