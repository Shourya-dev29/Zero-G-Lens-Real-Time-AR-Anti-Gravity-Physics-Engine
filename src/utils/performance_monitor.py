"""
performance_monitor.py: Real-time performance telemetry for Zero-G Lens.

Tracks per-frame latency, rolling FPS, P95 latency, and object counts.
All metrics use a sliding window to avoid unbounded memory growth.
"""
import time
import numpy as np
from collections import deque
from typing import Dict, Any


class PerformanceMonitor:
    def __init__(self, window: int = 60):
        """
        Args:
            window: Number of recent frames to include in rolling statistics.
        """
        self.window          = window
        self.latencies       = deque(maxlen=window)    # ms per frame
        self.timestamps      = deque(maxlen=window)    # wall-clock timestamps
        self.object_counts   = deque(maxlen=window)    # detected objects per frame
        self._npu_active     = True

    def record(self, latency_ms: float, num_objects: int = 0):
        """Record metrics for one frame."""
        self.latencies.append(latency_ms)
        self.timestamps.append(time.perf_counter())
        self.object_counts.append(num_objects)

    def stats(self) -> Dict[str, Any]:
        """Return current rolling statistics dict (safe to call every frame)."""
        if len(self.latencies) < 2:
            return {
                "latency_ms": 0.0,
                "fps":        0.0,
                "p95_ms":     0.0,
                "num_objects": 0,
                "npu_active":  self._npu_active,
            }

        lats   = np.array(self.latencies, dtype=np.float32)
        times  = list(self.timestamps)

        # Rolling FPS from actual wall-clock timestamps
        elapsed = times[-1] - times[0]
        fps     = (len(times) - 1) / elapsed if elapsed > 0 else 0.0

        return {
            "latency_ms":  float(lats[-1]),             # Last frame latency
            "avg_ms":      float(np.mean(lats)),
            "p95_ms":      float(np.percentile(lats, 95)),
            "fps":         round(fps, 1),
            "num_objects": int(self.object_counts[-1]) if self.object_counts else 0,
            "npu_active":  self._npu_active,
        }

    def set_npu_active(self, active: bool):
        self._npu_active = active

    def report(self):
        """Print a summary report to stdout (call at shutdown)."""
        if not self.latencies:
            return
        lats = np.array(self.latencies)
        print("\n── Performance Report ────────────────────────")
        print(f"  Frames recorded :  {len(lats)}")
        print(f"  Avg latency     :  {np.mean(lats):.1f} ms")
        print(f"  P50 latency     :  {np.percentile(lats, 50):.1f} ms")
        print(f"  P95 latency     :  {np.percentile(lats, 95):.1f} ms")
        print(f"  P99 latency     :  {np.percentile(lats, 99):.1f} ms")
        print(f"  Min / Max       :  {lats.min():.1f} / {lats.max():.1f} ms")
        print("──────────────────────────────────────────────\n")
