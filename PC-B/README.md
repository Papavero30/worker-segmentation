# PC-B Worker (RTX 5080 x1)

Node ini menjalankan 1 container worker untuk skenario homogen.

## Karakteristik

- Node: PC-B
- GPU: 1x RTX 5080
- Queue mode: homogen (+ legacy fallback)
- Metric port: 8001

## Konfigurasi

File utama:

- .env
- docker-compose.yml

Catatan implementasi:

- .env PC-B dipertahankan (sesuai constraint implementasi).
- Variabel queue/mode terbaru disuntik melalui docker-compose.yml:
  - CLUSTER_MODES=homogen
  - GPU_TYPE=RTX5080
  - QUEUE_HOMOGEN/QUEUE_HETEROGEN/QUEUE_LEGACY

## Jalankan

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-B"
docker compose up -d --build
```

## Validasi Cepat

```powershell
curl http://localhost:8001/health
curl http://localhost:8001/metrics
```

## Catatan

- Node ini dipakai sebagai pasangan PC-A untuk mode homogen.
- Pastikan model file tersedia: best_metric_model.pth.
