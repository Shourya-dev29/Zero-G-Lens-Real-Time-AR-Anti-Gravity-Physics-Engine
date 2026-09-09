"""
benchmark.py: Performance benchmarking tool for Zero-G Lens.

Measures end-to-end pipeline latency across N frames and produces a
detailed performance report, including per-component breakdown.

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --frames 300 --cam 0 --cpu-fallback
"""
import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.inference_engine import InferencePipeline


def parse_args():
    p = argparse.ArgumentParser(description="Zero-G Lens Benchmark")
    p.add_argument("--frames",       type=int,  default=200,   help="Number of benchmark frames")
    p.add_argument("--cpu-fallback", action="store_true",       help="Use CPU instead of NPU")
    p.add_argument("--resolution",   type=str,  default="1280x720", help="Simulated frame resolution WxH")
    return p.parse_args()


def run_benchmark(args):
    w, h  = map(int, args.resolution.split("x"))
    mode  = "CPU Fallback" if args.cpu_fallback else "QNN Hexagon HTP"

    print("="*60)
    print(f"  Zero-G Lens — NPU Benchmark")
    print("="*60)
    print(f"  Mode:       {mode}")
    print(f"  Frames:     {args.frames}")
    print(f"  Resolution: {w}×{h}")
    print("="*60)

    # Synthetic random frame (simulates webcam input)
    dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    pipeline = InferencePipeline(use_npu=not args.cpu_fallback, conf_thresh=0.4)
    pipeline.load_all_models()

    # ── Warm-up ──────────────────────────────────────────────────────
    print("\n[BENCH] Warming up (5 frames)...")
    for _ in range(5):
        pipeline.run_concurrent(dummy_frame)

    # ── Benchmark ────────────────────────────────────────────────────
    print(f"[BENCH] Running {args.frames} frames...\n")
    component_times = {"yolo": [], "midas": [], "hand": [], "total": []}

    for i in range(args.frames):
        t0 = time.perf_counter()

        # Individual timing (run serially for component breakdown)
        t_yolo_0 = time.perf_counter()
        pipeline._run_yolo(dummy_frame)
        component_times["yolo"].append((time.perf_counter() - t_yolo_0) * 1000)

        t_midas_0 = time.perf_counter()
        pipeline._run_midas(dummy_frame)
        component_times["midas"].append((time.perf_counter() - t_midas_0) * 1000)

        t_hand_0 = time.perf_counter()
        pipeline._run_hand(dummy_frame)
        component_times["hand"].append((time.perf_counter() - t_hand_0) * 1000)

        component_times["total"].append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 50 == 0:
            recent_total = np.mean(component_times["total"][-50:])
            print(f"  Frame {i+1:>4}/{args.frames}  |  Avg (last 50): {recent_total:.1f} ms")

    # ── Report ────────────────────────────────────────────────────────
    print("\n── Benchmark Results ─────────────────────────────────────")
    print(f"  {'Component':<15} {'Mean':>8} {'P50':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for key in ["yolo", "midas", "hand", "total"]:
        times = np.array(component_times[key])
        label = "YOLOv8n" if key == "yolo" else \
                "MiDaS"   if key == "midas" else \
                "Hand Trk" if key == "hand" else "TOTAL (serial)"
        print(f"  {label:<15} {np.mean(times):>7.1f}ms {np.percentile(times,50):>7.1f}ms "
              f"{np.percentile(times,95):>7.1f}ms {times.min():>7.1f}ms {times.max():>7.1f}ms")

    total_arr = np.array(component_times["total"])
    concurrent_est = max(component_times["yolo"][-1],
                         component_times["midas"][-1],
                         component_times["hand"][-1])
    print(f"\n  Estimated concurrent latency (NPU parallel): ~{concurrent_est:.1f} ms")
    print(f"  Equivalent FPS (concurrent):                 ~{1000/concurrent_est:.1f}")
    print("──────────────────────────────────────────────────────────\n")

    pipeline.shutdown()


if __name__ == "__main__":
    run_benchmark(parse_args())
