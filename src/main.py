"""
Zero-G Lens — Main Entry Point
Snapdragon X Elite NPU-accelerated AR Anti-Gravity Physics Engine

Run:
    python src/main.py [options]

Options:
    --gravity FLOAT       Vertical gravity value (0 = zero-G, -9.8 = Earth). Default: 0
    --conf FLOAT          YOLO object detection confidence threshold. Default: 0.45
    --cpu-fallback        Disable QNN NPU and use CPU for inference (testing only)
    --high-contrast       Enable high-contrast accessibility mode
    --cam INT             Camera device index. Default: 0
"""
import argparse
import threading
import time
import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.inference_engine import InferencePipeline
from pipeline.camera import CameraCapture
from physics.physics_world import PhysicsWorld
from renderer.ar_overlay import ARRenderer
from utils.performance_monitor import PerformanceMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-G Lens AR Anti-Gravity Engine")
    parser.add_argument("--gravity",       type=float, default=0.0,  help="Vertical gravity (0=zero-G)")
    parser.add_argument("--conf",          type=float, default=0.45, help="YOLO detection confidence threshold")
    parser.add_argument("--cpu-fallback",  action="store_true",       help="Use CPU instead of NPU")
    parser.add_argument("--high-contrast", action="store_true",       help="High-contrast accessibility mode")
    parser.add_argument("--cam",           type=int,   default=0,    help="Camera device index")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("  Zero-G Lens | Snapdragon AI Lab Challenge")
    print("="*60)
    print(f"  Gravity:      {args.gravity} m/s²")
    print(f"  Conf thresh:  {args.conf}")
    print(f"  NPU mode:     {'CPU Fallback' if args.cpu_fallback else 'QNN Hexagon HTP'}")
    print(f"  Camera:       /dev/video{args.cam}")
    print("="*60)

    # ── Initialize subsystems ──────────────────────────────────────
    camera    = CameraCapture(device_id=args.cam, resolution=(1280, 720), fps=30)
    pipeline  = InferencePipeline(use_npu=not args.cpu_fallback, conf_thresh=args.conf)
    physics   = PhysicsWorld(gravity=args.gravity)
    renderer  = ARRenderer(high_contrast=args.high_contrast)
    perf_mon  = PerformanceMonitor(window=30)

    # Warm-up: load models onto NPU, prime VTCM cache
    print("\n[INIT] Loading AI models onto Hexagon HTP...")
    pipeline.load_all_models()
    print("[INIT] NPU warm-up complete. Launching capture...\n")

    # ── Shared state (thread-safe via lock) ────────────────────────
    result_lock  = threading.Lock()
    frame_ready  = threading.Event()
    running      = threading.Event()
    running.set()

    shared = {
        "frame":       None,
        "detections":  [],
        "depth_map":   None,
        "hand_data":   {},
    }

    # ── Camera capture thread (dedicated) ─────────────────────────
    def capture_loop():
        for frame in camera.stream():
            if not running.is_set():
                break
            with result_lock:
                shared["frame"] = frame
            frame_ready.set()

    # ── Main processing + rendering loop ──────────────────────────
    def process_loop():
        while running.is_set():
            if not frame_ready.wait(timeout=0.1):
                continue
            frame_ready.clear()

            with result_lock:
                frame = shared["frame"]

            if frame is None:
                continue

            t_start = time.perf_counter()

            # ── NPU Concurrent Inference ───────────────────────────
            detections, depth_map, hand_data = pipeline.run_concurrent(frame)

            # ── 3D Fusion: 2D bbox + depth → 3D world position ────
            scene_objects = pipeline.fuse(detections, depth_map)

            # ── Physics Step ───────────────────────────────────────
            physics.sync_objects(scene_objects)
            physics.apply_hand_forces(hand_data)
            physics.step()

            # ── Render AR overlay ──────────────────────────────────
            output = renderer.draw(frame, physics.get_state(), hand_data)

            # ── HUD ────────────────────────────────────────────────
            latency_ms = (time.perf_counter() - t_start) * 1000
            perf_mon.record(latency_ms, num_objects=len(scene_objects))
            renderer.draw_hud(output, perf_mon.stats())

            # ── Display ────────────────────────────────────────────
            try:
                renderer.show(output)
            except SystemExit:
                print("\n[EXIT] User quit. Shutting down...")
                running.clear()
                break

    # ── Start threads ──────────────────────────────────────────────
    t_capture = threading.Thread(target=capture_loop,  daemon=True, name="CaptureThread")
    t_process = threading.Thread(target=process_loop,  daemon=True, name="ProcessThread")

    t_capture.start()
    t_process.start()

    try:
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt. Shutting down...")
        running.clear()

    camera.release()
    pipeline.shutdown()
    physics.shutdown()
    print("[EXIT] Goodbye from Zero-G Lens.")


if __name__ == "__main__":
    main()
