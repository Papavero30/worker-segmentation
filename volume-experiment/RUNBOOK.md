# RUNBOOK — Eksperimen Distributed S2 & S3 (Volume-Level, nnU-Net ADAM)

Melanjutkan S1 (laptop saja, sudah selesai: wall-clock 2331.6 s, throughput 0.77 scan/menit).
Sekarang:
- **S2** = Laptop + PC-A
- **S3** = Laptop + PC-A + PC-B

Worker memakai **`volume_worker.py`** (volume-level, SAMA dengan S1 → speedup valid).
Bukan worker chunk lama di `PC-A/`,`PC-B/` (itu menghasilkan mask kosong dgn nnU-Net).

Identitas tiap node (via env var):
| Node | WORKER_NAME | GPU_TYPE | METRICS_PORT |
|---|---|---|---|
| Laptop | master-3050ti | rtx-3050ti-mobile | 8000 |
| PC-A | pc-a-5080 | rtx-5080 | 8001 |
| PC-B | pc-b-5080 | rtx-5080 | 8001 |

Master (broker) Tailscale IP: **100.110.113.24** · RabbitMQ **5674** · Redis **6381**
Kredensial: user `brainnav` / pass `BrainNav_Secure_2025!` / vhost `brainnav_vhost`.

---

## FASE 0 — Distribusi kode & aset (sekali)

### 0.1 Laptop: push kode
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation"
git add volume-experiment
git commit -m "Add volume-level distributed experiment harness"
git push origin main
```

### 0.2 PC-A & PC-B: pull kode (via RDP)
```powershell
cd <path>\worker-segmentation
git pull origin main
# kode ada di: worker-segmentation\volume-experiment\
```

### 0.3 Model + scan ke tiap PC (JANGAN lewat git — terlalu besar)
Copy via RDP (clipboard/shared drive) ke tiap PC:
- folder model **`nnUnet-Papavero\`** (checkpoint 5-fold)
- scan **`10070B.nii.gz`** → taruh di `worker-segmentation\volume-experiment\scans\10070B.nii.gz`

> Catatan: scan sumber di laptop = `10070B\10070B\pre\TOF.nii.gz` (rename jadi `10070B.nii.gz`).

---

## FASE 1 — Laptop (master) prep

### 1.1 Nyalakan Docker Desktop, lalu broker + monitoring
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\Master-Segmentation"
docker compose up -d rabbitmq redis
docker compose up -d --no-deps prometheus grafana
docker ps   # rabbitmq(5674), redis(6381), prometheus(9091), grafana(3005) harus Up
```

### 1.2 Siapkan scan + folder mask di laptop
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\volume-experiment"
mkdir scans -Force
Copy-Item "..\..\10070B\10070B\pre\TOF.nii.gz" scans\10070B.nii.gz
mkdir masks -Force
```

### 1.3 Start worker laptop (biarkan terminal ini jalan)
```powershell
$env:RABBITMQ_HOST="localhost"; $env:RABBITMQ_PORT="5674"
$env:RABBITMQ_USER="brainnav"; $env:RABBITMQ_PASS="BrainNav_Secure_2025!"; $env:RABBITMQ_VHOST="brainnav_vhost"
$env:REDIS_URL="redis://:BrainNav_Secure_2025!@localhost:6381/0"
$env:NNUNET_MODEL_FOLDER="..\..\nnUnet-Papavero"; $env:PAYLOAD_MODE="reference"; $env:SCAN_DIR=".\scans"
$env:SAVE_MASK_DIR=".\masks"
$env:WORKER_NAME="master-3050ti"; $env:GPU_TYPE="rtx-3050ti-mobile"; $env:METRICS_PORT="8000"
..\..\nnunet_gpu_env\Scripts\python.exe volume_worker.py
```
Siap jika: `Worker master-3050ti ... menunggu task`.

---

## FASE 2 — S2 (Laptop + PC-A)

### 2.1 PC-A: buat venv (sekali) — di `worker-segmentation\volume-experiment`
```powershell
py -3.11 -m venv gpu_env
.\gpu_env\Scripts\python -m pip install --upgrade pip
.\gpu_env\Scripts\pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\gpu_env\Scripts\pip install nnunetv2==2.5.2 --no-deps
.\gpu_env\Scripts\pip install "numpy==1.26.4" acvl-utils batchgenerators batchgeneratorsv2 "dynamic-network-architectures==0.4.3" einops graphviz imagecodecs matplotlib nibabel pandas requests scikit-image scikit-learn scipy seaborn SimpleITK tifffile tqdm yacs blosc2 msgpack ndindex py-cpuinfo pika redis flask prometheus-client tenacity
.\gpu_env\Scripts\python -c "import torch,nnunetv2;print('cuda',torch.cuda.is_available())"   # True
```

### 2.2 PC-A: verifikasi koneksi ke master
```powershell
Test-NetConnection 100.110.113.24 -Port 5674   # True
Test-NetConnection 100.110.113.24 -Port 6381   # True
```

### 2.3 PC-A: start worker (biarkan jalan). Sesuaikan path model/scan PC.
```powershell
$env:RABBITMQ_HOST="100.110.113.24"; $env:RABBITMQ_PORT="5674"
$env:RABBITMQ_USER="brainnav"; $env:RABBITMQ_PASS="BrainNav_Secure_2025!"; $env:RABBITMQ_VHOST="brainnav_vhost"
$env:REDIS_URL="redis://:BrainNav_Secure_2025!@100.110.113.24:6381/0"
$env:NNUNET_MODEL_FOLDER="<path ke nnUnet-Papavero>"; $env:PAYLOAD_MODE="reference"; $env:SCAN_DIR=".\scans"
$env:SAVE_MASK_DIR=".\masks"
$env:WORKER_NAME="pc-a-5080"; $env:GPU_TYPE="rtx-5080"; $env:METRICS_PORT="8001"
.\gpu_env\Scripts\python volume_worker.py
```

### 2.4 Laptop (terminal baru): publish + collect S2
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\worker-segmentation\volume-experiment"
$PY = "..\..\nnunet_gpu_env\Scripts\python.exe"
$RU = "redis://:BrainNav_Secure_2025!@localhost:6381/0"
& $PY publish_tasks.py --scan-id 10070B --count 30 --scale S2 --run-id S2_full --rabbitmq-host localhost --redis-url $RU
& $PY collect_results.py --run-id S2_full --count 30 --out S2.csv --baseline-csv S1.csv --nodes 2 --redis-url $RU
```
> Pastikan `S1.csv` ada di folder ini (copy dari `distributed-experiment\S1.csv`) agar speedup terhitung.

Collector mencetak: wall-clock, throughput, **GRAFANA time range (start→end)**, E2E latency, distribusi beban (master vs pc-a), **speedup & efficiency**.

---

## FASE 3 — S3 (Laptop + PC-A + PC-B)

PC-A & laptop tetap jalan. Tambahkan PC-B:

### 3.1 PC-B: venv (sekali) — sama persis dgn 2.1
### 3.2 PC-B: start worker — sama dgn 2.3, ganti identitas:
```powershell
$env:WORKER_NAME="pc-b-5080"; $env:GPU_TYPE="rtx-5080"; $env:METRICS_PORT="8001"
# (env lain sama dgn PC-A)
.\gpu_env\Scripts\python volume_worker.py
```

### 3.3 Laptop: publish + collect S3
```powershell
& $PY publish_tasks.py --scan-id 10070B --count 30 --scale S3 --run-id S3_full --rabbitmq-host localhost --redis-url $RU
& $PY collect_results.py --run-id S3_full --count 30 --out S3.csv --baseline-csv S1.csv --nodes 3 --redis-url $RU
```

---

## FASE 4 — Mencatat Grafana (kapan & apa)

### Kapan (time range)
`collect_results.py` mencetak baris:
```
 GRAFANA time range  : 2026-06-xx HH:MM:SS  ->  2026-06-xx HH:MM:SS  (waktu lokal)
```
Juga tersimpan di CSV (`# t_start`, `# t_end`). **Itulah rentang yang harus Anda set di Grafana.**

### Cara set di Grafana (`http://localhost:3005`, admin/admin)
1. Pojok kanan atas → picker waktu → **Absolute time range**.
2. Isi From/To dengan `t_start`/`t_end` dari output (beri buffer ±1 menit).
3. Set refresh "Off" (biar statis saat screenshot).

### Apa yang di-screenshot (Explore → datasource Prometheus)
| Panel | Query |
|---|---|
| Throughput total | `sum(rate(volseg_tasks_processed_total[1m]))*60` |
| Throughput per worker | `sum by (worker_name) (rate(volseg_tasks_processed_total[1m]))*60` |
| Distribusi beban | `sum by (worker_name) (volseg_tasks_processed_total)` |
| Task aktif (paralelisme) | `volseg_active_tasks` |
| Waktu inferensi mean | `rate(volseg_inference_seconds_sum[2m]) / rate(volseg_inference_seconds_count[2m])` |
| Queue wait mean | `rate(volseg_queue_wait_seconds_sum[2m]) / rate(volseg_queue_wait_seconds_count[2m])` |

Screenshot tiap config (S2, S3) pada rentang waktunya masing-masing → bukti visual scaling & fair dispatch (5080 ambil lebih banyak task daripada 3050Ti).

> Catatan: angka FINAL paper ambil dari CSV (`collect_results.py`); Grafana untuk figure visual.
> Prometheus men-scrape worker laptop via `host.docker.internal:8000`. Untuk men-scrape PC-A/PC-B
> (port 8001) Prometheus perlu menjangkau IP PC — kalau tidak terjangkau, andalkan CSV untuk
> distribusi beban (collector tetap mencatatnya dari Redis).

---

## FASE 5 — Perbandingan gambar segmentasi (slice sama)

Tujuan: buktikan distributed menghasilkan segmentasi **identik** dengan single-node
(tidak menurunkan kualitas), pada slice 24 & 25 subjek 10070B (sama dgn paper).

### 5.1 Kumpulkan mask
Worker menyimpan 1 mask/run/worker di folder `masks\` masing-masing node:
- Laptop: `masks\S2_full__master-3050ti__10070B.nii.gz`
- PC-A: `masks\S2_full__pc-a-5080__10070B.nii.gz`  → **copy ke laptop** (RDP)
- (S3) PC-B: `masks\S3_full__pc-b-5080__10070B.nii.gz` → copy ke laptop

### 5.2 Render perbandingan (di laptop) pakai script existing
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi"
$PY = ".\nnunet_gpu_env\Scripts\python.exe"
# contoh: mask dari PC-A (RTX 5080)
& $PY worker-segmentation\scripts_nnunet\visualize_segmentation.py `
  --tof "10070B\10070B\pre\TOF.nii.gz" `
  --gt  "10070B\10070B\aneurysms.nii.gz" `
  --pred "worker-segmentation\volume-experiment\masks\S2_full__pc-a-5080__10070B.nii.gz" `
  --out-dir ".\hasil_distributed_pcA"
```
Hasil: `hasil_distributed_pcA\compare_slice_024.png`, `compare_slice_025.png`.

### 5.3 Bandingkan
- Referensi single-node (laptop, S1/benchmark): `hasil_gpu_bench\prediksi_mask.nii.gz`
  atau gambar paper `hasil_adam_10070B\compare\compare_slice_024.png` / `025.png`.
- Bandingkan slice 24 & 25 hasil PC-A (dan PC-B) vs single-node.
- **Ekspektasi**: identik / nyaris identik (model & input sama, deterministik) → Dice tetap
  ~0.1296, lokasi di Circle of Willis. Ini bukti distributed tidak mengubah hasil segmentasi.

---

## Troubleshooting cepat
| Gejala | Solusi |
|---|---|
| Worker `ProbableAuthenticationError` (RabbitMQ) | pass harus `BrainNav_Secure_2025!` (sudah di-set di broker) |
| `connecting to ...6381 refused` | Docker/Redis belum jalan di laptop (FASE 1.1) |
| `Test-NetConnection ...5674` False di PC | tunnel/Tailscale belum jalan, atau firewall laptop (rule inbound Private 5674/6381 sudah dibuat) |
| Semua task ke 1 worker | worker lain belum running / belum subscribe; pastikan ketiganya menyala sebelum publish |
| Mask tidak tersimpan | `SAVE_MASK_DIR` belum di-set di env worker |
