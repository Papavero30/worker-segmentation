#!/usr/bin/env python3
"""
Publisher Task Volume (tanpa BrainNav, tanpa auth)
==================================================

Enqueue N task ke RabbitMQ queue volume-level. Tiap task membawa SATU volume
TOF (replikasi scan yang sama N kali sebagai workload homogen untuk benchmark
distributed). Mode payload:
  - base64    : isi file .nii.gz di-embed base64 di dalam message (default,
                sesuai konvensi proyek). Message besar (~120MB/scan 90MB).
  - reference : message hanya membawa scan_id; worker baca file lokal.

Mencatat run_id + t_start ke Redis supaya collect_results.py bisa menghitung
wall-clock throughput.

CONTOH
------
  python publish_tasks.py --scan ../10070B/10070B/pre/TOF.nii.gz \
      --scan-id 10070B --count 30 --scale S3

  # mode reference (scan harus sudah ada di SCAN_DIR tiap worker):
  python publish_tasks.py --scan-id 10070B --count 30 --scale S2 --mode reference
"""
import argparse
import base64
import json
import os
import sys
import time
import uuid

try:
    import pika
    import redis
except ImportError as e:
    print(f"[ERROR] butuh pika + redis: {e}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish volume segmentation tasks")
    ap.add_argument("--scan", help="Path file .nii.gz (wajib untuk mode base64)")
    ap.add_argument("--scan-id", required=True, help="ID scan (label), mis. 10070B")
    ap.add_argument("--count", type=int, default=30, help="Jumlah task (default 30)")
    ap.add_argument("--scale", default="S?", help="Label konfigurasi: S1/S2/S3")
    ap.add_argument("--mode", default="reference", choices=["base64", "reference"])
    ap.add_argument("--queue", default=os.getenv("VOLUME_QUEUE", "segmentation_volume_tasks"))
    ap.add_argument("--rabbitmq-host", default=os.getenv("RABBITMQ_HOST", "100.110.113.24"))
    ap.add_argument("--rabbitmq-port", type=int, default=int(os.getenv("RABBITMQ_PORT", "5674")))
    ap.add_argument("--rabbitmq-user", default=os.getenv("RABBITMQ_USER", "brainnav"))
    ap.add_argument("--rabbitmq-pass", default=os.getenv("RABBITMQ_PASS", "BrainNav_Secure_2025!"))
    ap.add_argument("--rabbitmq-vhost", default=os.getenv("RABBITMQ_VHOST", "brainnav_vhost"))
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://:BrainNav_Secure_2025!@100.110.113.24:6381/0"))
    ap.add_argument("--run-id", default=None, help="Override run_id (default auto)")
    args = ap.parse_args()

    run_id = args.run_id or f"{args.scale}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    volume_b64 = None
    if args.mode == "base64":
        if not args.scan or not os.path.isfile(args.scan):
            ap.error("mode base64 butuh --scan path file .nii.gz yang valid")
        size_mb = os.path.getsize(args.scan) / 1e6
        print(f"Encoding {args.scan} ({size_mb:.1f} MB) -> base64 ...")
        with open(args.scan, "rb") as f:
            volume_b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"  base64 size: {len(volume_b64)/1e6:.1f} MB per message")
        if len(volume_b64) > 128 * 1024 * 1024:
            print("  [WARN] message > 128 MiB; naikkan RabbitMQ max_message_size!")

    # Redis: simpan metadata run (untuk collector)
    r = redis.Redis.from_url(args.redis_url)
    r.ping()
    # reset counter selesai untuk run ini
    r.delete(f"segmentation:exp:done:{run_id}")

    # RabbitMQ connect
    creds = pika.PlainCredentials(args.rabbitmq_user, args.rabbitmq_pass)
    params = pika.ConnectionParameters(
        host=args.rabbitmq_host, port=args.rabbitmq_port, virtual_host=args.rabbitmq_vhost,
        credentials=creds, heartbeat=900, blocked_connection_timeout=600,
    )
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=args.queue, durable=True, arguments={"x-message-ttl": 3600000})

    print(f"\nRun ID   : {run_id}")
    print(f"Scale    : {args.scale}")
    print(f"Queue    : {args.queue}  @ {args.rabbitmq_host}:{args.rabbitmq_port}")
    print(f"Mode     : {args.mode}")
    print(f"Count    : {args.count}\n")

    t_start_ns = time.time_ns()
    r.setex(f"segmentation:exp:meta:{run_id}", 86400, json.dumps({
        "run_id": run_id, "scale": args.scale, "scan_id": args.scan_id,
        "count": args.count, "mode": args.mode, "t_start_ns": t_start_ns,
    }))

    for i in range(args.count):
        task = {
            "task_id": f"{run_id}_t{i:03d}",
            "run_id": run_id,
            "scan_id": args.scan_id,
            "idx": i, "total": args.count,
            "scale": args.scale,
            "retry_count": 0,
            "published_at_ns": time.time_ns(),
        }
        if args.mode == "base64":
            task["volume_b64"] = volume_b64

        ch.basic_publish(
            exchange="", routing_key=args.queue,
            body=json.dumps(task),
            properties=pika.BasicProperties(delivery_mode=2),  # persistent
        )
        print(f"  [{i+1}/{args.count}] published task {task['task_id']}")

    conn.close()
    print(f"\nSelesai enqueue {args.count} task. run_id={run_id}")
    print(f"Lanjut: python collect_results.py --run-id {run_id} --count {args.count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
