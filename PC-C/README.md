# PC-C Worker Template (RTX 2080 x1)

Node ini adalah template worker untuk skenario heterogen.

## Status

- Template sudah siap di repository.
- Deploy fisik dilakukan saat PC-C tersedia.

## Karakteristik

- Node: PC-C
- GPU: 1x RTX 2080
- Queue mode: heterogen (+ legacy fallback)
- Metric port: 8001

## Konfigurasi

File utama:

- .env
- docker-compose.yml

Variable penting:

- CLUSTER_MODES=heterogen
- GPU_TYPE=RTX2080
- QUEUE_HOMOGEN=segmentation_tasks_homogen
- QUEUE_HETEROGEN=segmentation_tasks_heterogen
- QUEUE_LEGACY=segmentation_tasks

## Jalankan (saat node siap)

```powershell
Set-Location "d:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\PC-C"
docker compose up -d --build
```

## Checklist Aktivasi Node

1. Salin best_metric_model.pth ke folder PC-C.
2. Verifikasi akses ke master (RabbitMQ/Redis).
3. Start container.
4. Verifikasi /health dan /metrics.

## Catatan

- Node ini digunakan sebagai pasangan PC-A pada mode heterogen.
- Jika IP PC-C final berbeda, sesuaikan konfigurasi monitoring master (prometheus.yml).
