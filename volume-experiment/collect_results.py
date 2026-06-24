#!/usr/bin/env python3
"""
Collector Hasil & Metrik Eksperimen Distributed
===============================================

Menunggu N task selesai (counter Redis), lalu membaca semua log per-task,
menulis CSV, dan menghitung metrik agregat untuk paper:
  - wall-clock batch (t_end - t_start)
  - throughput (task/menit)
  - end-to-end latency per task (mean / p50 / p95)
  - inference time per task (mean) -- komputasi murni
  - queue wait (mean) -- transfer + antri (communication overhead)
  - distribusi beban per worker (berapa task/worker)
  - speedup & efficiency (jika file baseline S1 diberikan)

CONTOH
------
  # S1 (baseline)
  python collect_results.py --run-id S1_2026... --count 30 --out S1.csv

  # S3, hitung speedup vs S1
  python collect_results.py --run-id S3_2026... --count 30 --out S3.csv \
      --baseline-csv S1.csv --nodes 3
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time

try:
    import redis
except ImportError as e:
    print(f"[ERROR] butuh redis: {e}")
    sys.exit(1)


def pctile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def read_baseline_wallclock(path):
    """Ambil wall-clock detik dari baris '# wallclock_sec,<value>' di CSV baseline."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# wallclock_sec,"):
                try:
                    return float(line.strip().split(",", 1)[1])
                except (ValueError, IndexError):
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect distributed experiment results")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", default=None, help="CSV output (default <run-id>.csv)")
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://:BrainNav_Secure_2025!@100.110.113.24:6381/0"))
    ap.add_argument("--timeout", type=int, default=7200, help="Detik menunggu semua task")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--baseline-csv", default=None, help="CSV S1 untuk hitung speedup")
    ap.add_argument("--nodes", type=int, default=1, help="Jumlah node aktif (untuk efficiency)")
    args = ap.parse_args()

    out_path = args.out or f"{args.run_id}.csv"
    r = redis.Redis.from_url(args.redis_url)
    r.ping()

    meta_raw = r.get(f"segmentation:exp:meta:{args.run_id}")
    meta = json.loads(meta_raw) if meta_raw else {}
    t_start_ns = int(meta.get("t_start_ns", 0))

    # Tunggu sampai semua task selesai
    print(f"Menunggu {args.count} task selesai (run_id={args.run_id})...")
    start_wait = time.time()
    while True:
        done = int(r.get(f"segmentation:exp:done:{args.run_id}") or 0)
        print(f"  selesai {done}/{args.count}", end="\r")
        if done >= args.count:
            break
        if time.time() - start_wait > args.timeout:
            print(f"\n[WARN] timeout; lanjut dengan {done}/{args.count} task.")
            break
        time.sleep(args.poll)
    print()

    # Baca semua log
    rows = []
    for key in r.scan_iter(match=f"segmentation:exp:log:{args.run_id}:*", count=200):
        raw = r.get(key)
        if raw:
            rows.append(json.loads(raw))
    rows.sort(key=lambda x: x.get("idx", 0))

    if not rows:
        print("[ERROR] tidak ada log ditemukan. Pastikan worker menulis ke Redis yang sama.")
        return 2

    done_ns = [x["done_ns"] for x in rows]
    t_end_ns = max(done_ns)
    if t_start_ns == 0:
        t_start_ns = min(x["published_at_ns"] for x in rows)
    wallclock_sec = (t_end_ns - t_start_ns) / 1e9

    e2e_ms = [(x["done_ns"] - x["published_at_ns"]) / 1e6 for x in rows]
    infer_ms = [x["inference_ms"] for x in rows]
    qwait_ms = [x["queue_wait_ms"] for x in rows]

    # Distribusi beban per worker
    per_worker = {}
    for x in rows:
        per_worker.setdefault(x["worker_name"], 0)
        per_worker[x["worker_name"]] += 1

    throughput_per_min = len(rows) / (wallclock_sec / 60.0) if wallclock_sec > 0 else 0.0

    # Rentang waktu absolut (untuk set time range di Grafana/Prometheus)
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start_ns / 1e9))
    t_end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end_ns / 1e9))

    # CSV per-task
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "idx", "worker_name", "gpu_type",
                    "queue_wait_ms", "inference_ms", "total_ms",
                    "e2e_ms", "fg_voxels"])
        for x in rows:
            e2e = (x["done_ns"] - x["published_at_ns"]) / 1e6
            w.writerow([x["task_id"], x["idx"], x["worker_name"], x["gpu_type"],
                        x["queue_wait_ms"], x["inference_ms"], x["total_ms"],
                        f"{e2e:.0f}", x["fg_voxels"]])
        # ringkasan sebagai komentar (dibaca oleh --baseline-csv)
        f.write(f"# wallclock_sec,{wallclock_sec:.2f}\n")
        f.write(f"# throughput_per_min,{throughput_per_min:.3f}\n")
        f.write(f"# t_start,{t_start_str}\n")
        f.write(f"# t_end,{t_end_str}\n")

    # Speedup vs baseline
    speedup = efficiency = None
    base_wall = read_baseline_wallclock(args.baseline_csv)
    if base_wall and wallclock_sec > 0:
        speedup = base_wall / wallclock_sec
        efficiency = speedup / max(1, args.nodes)

    # Cetak ringkasan
    print("=" * 60)
    print(f" RINGKASAN  run_id={args.run_id}  scale={meta.get('scale','?')}")
    print("=" * 60)
    print(f" Task selesai        : {len(rows)}/{args.count}")
    print(f" Wall-clock batch    : {wallclock_sec:.1f} s")
    print(f" GRAFANA time range  : {t_start_str}  ->  {t_end_str}  (waktu lokal)")
    print(f" Throughput          : {throughput_per_min:.2f} task/menit")
    print(f" E2E latency  mean   : {st.mean(e2e_ms)/1000:.1f} s")
    print(f"              p50    : {pctile(e2e_ms,0.5)/1000:.1f} s")
    print(f"              p95    : {pctile(e2e_ms,0.95)/1000:.1f} s")
    print(f" Inference   mean    : {st.mean(infer_ms)/1000:.1f} s  (komputasi)")
    print(f" Queue wait  mean    : {st.mean(qwait_ms)/1000:.1f} s  (transfer+antri)")
    print(f" Distribusi beban    :")
    for wk, n in sorted(per_worker.items()):
        print(f"     {wk:24s}: {n} task")
    if speedup is not None:
        print(f" Speedup vs baseline : {speedup:.2f}x")
        print(f" Efficiency ({args.nodes} node): {efficiency:.2f}")
    print("=" * 60)
    print(f" CSV: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
