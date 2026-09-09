"""
InferencePipeline: Manages three concurrent QNN NPU inference sessions.

Architecture:
  - Each model runs in its own thread via ThreadPoolExecutor.
  - QNN Execution Provider (Hexagon HTP) handles true parallel execution
    on independent Hexagon DSP cores — not time-sliced on the CPU.
  - VTCM weight pinning eliminates DDR bandwidth bottleneck for small models.

NPU Target: Snapdragon X Elite Hexagon HTP (QNN EP via ONNX Runtime)
Fallback:   CPUExecutionProvider (for development on non-Snapdragon hardware)
"""
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.object_detector import preprocess_yolo, postprocess_yolo
from pipeline.depth_estimator import preprocess_midas, postprocess_midas
from pipeline.hand_tracker import preprocess_hand, postprocess_hand


class InferencePipeline:
    # ── Model file paths ────────────────────────────────────────────
    MODEL_DIR = Path(__file__).parents[2] / "models" / "qnn"
    MODEL_PATHS = {
        "yolo":  "yolov8n.onnx",
        "midas": "midas_v21_small.onnx",
        "hand":  "mediapipe_hand.onnx",
    }

    # ── QNN Execution Provider options for Hexagon HTP ──────────────
    QNN_OPTIONS = {
        "backend_path":              "QnnHtp.dll",   # Hexagon HTP backend DLL
        "enable_htp_fp16_precision": "1",            # Use FP16 on NPU (2× throughput)
        "htp_performance_mode":      "burst",        # Maximum NPU clock frequency
        "vtcm_mb":                   "8",            # Pin weights to 8MB on-chip VTCM
        "qnn_context_priority":      "high",         # High QNN scheduling priority
    }

    def __init__(self, use_npu: bool = True, conf_thresh: float = 0.45):
        self.use_npu     = use_npu
        self.conf_thresh = conf_thresh
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.executor    = ThreadPoolExecutor(max_workers=3, thread_name_prefix="NPU")

    # ── Session Initialization ──────────────────────────────────────
    def load_all_models(self):
        """Load and warm-up all QNN inference sessions on the Hexagon NPU."""
        if self.use_npu:
            providers = [
                ("QNNExecutionProvider", self.QNN_OPTIONS),
                "CPUExecutionProvider",
            ]
            print("[NPU] Target: Hexagon HTP (QNN Execution Provider)")
        else:
            providers = ["CPUExecutionProvider"]
            print("[NPU] Target: CPU (fallback mode)")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_mem_pattern       = True
        sess_opts.enable_cpu_mem_arena     = True

        for name, filename in self.MODEL_PATHS.items():
            model_path = self.MODEL_DIR / filename
            if not model_path.exists():
                print(f"[WARN] Model not found: {model_path}. Run scripts/setup_models.py first.")
                continue
            self.sessions[name] = ort.InferenceSession(
                str(model_path), sess_opts, providers=providers
            )
            self._warmup(name)
            print(f"[NPU] ✅ {name:<8} loaded → {filename}")

    def _warmup(self, name: str, n_iters: int = 3):
        """Run dummy inference to prime the NPU context and VTCM cache."""
        session = self.sessions[name]
        inp     = session.get_inputs()[0]
        shape   = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dummy   = np.zeros(shape, dtype=np.float32)
        for _ in range(n_iters):
            session.run(None, {inp.name: dummy})

    def shutdown(self):
        self.executor.shutdown(wait=False)

    # ── Individual Inference Methods ────────────────────────────────
    def _run_yolo(self, frame: np.ndarray) -> list:
        """
        YOLOv8n object detection.
        Returns: list of (class_id: int, confidence: float, bbox_xyxy: tuple)
        """
        if "yolo" not in self.sessions:
            return []
        inp     = preprocess_yolo(frame, size=640)
        output  = self.sessions["yolo"].run(None, {"images": inp})[0]
        return postprocess_yolo(
            output,
            orig_shape=frame.shape[:2],
            conf_thresh=self.conf_thresh,
            iou_thresh=0.45,
        )

    def _run_midas(self, frame: np.ndarray) -> np.ndarray:
        """
        MiDaS v2.1 depth estimation.
        Returns: (H, W) float32 normalized inverse depth map in range [0, 1].
        """
        if "midas" not in self.sessions:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.float32)
        inp       = preprocess_midas(frame, size=256)
        depth_raw = self.sessions["midas"].run(None, {"input": inp})[0]
        return postprocess_midas(depth_raw, orig_shape=frame.shape[:2])

    def _run_hand(self, frame: np.ndarray) -> dict:
        """
        MediaPipe hand landmark detection.
        Returns: dict with 'hands' list, each containing landmarks, gesture, wrist_pos.
        """
        if "hand" not in self.sessions:
            return {"hands": []}
        inp = preprocess_hand(frame, size=256)
        raw = self.sessions["hand"].run(None, {"input_1": inp})
        return postprocess_hand(raw, frame_shape=frame.shape[:2])

    # ── Concurrent Inference ────────────────────────────────────────
    def run_concurrent(self, frame: np.ndarray):
        """
        Submit all three inference tasks concurrently to the ThreadPoolExecutor.
        On Snapdragon X Elite, QNN HTP schedules these on independent Hexagon DSP
        cores — achieving true parallelism, not CPU thread time-slicing.

        Returns: (detections, depth_map, hand_data)
        """
        futures = {
            self.executor.submit(self._run_yolo,  frame): "yolo",
            self.executor.submit(self._run_midas, frame): "midas",
            self.executor.submit(self._run_hand,  frame): "hand",
        }
        results = {"yolo": [], "midas": None, "hand": {"hands": []}}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[WARN] Inference error in {key}: {e}")
        return results["yolo"], results["midas"], results["hand"]

    # ── Result Fusion ───────────────────────────────────────────────
    def fuse(self, detections: list, depth_map: np.ndarray) -> list:
        """
        Fuse 2D object bounding boxes with per-pixel depth map to produce
        3D world-space positions for each detected object.

        World coordinate system:
          X: [-1, +1]  left → right
          Y: [-1, +1]  bottom → top  (flipped from image coords)
          Z: [0,  1]   inverse depth (0=far, 1=near)

        Returns: list of scene object dicts.
        """
        if depth_map is None:
            depth_map = np.zeros((1, 1), dtype=np.float32)

        h, w = depth_map.shape
        scene_objects = []

        for det in detections:
            class_id, confidence, (x1, y1, x2, y2) = det
            cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, h - 1))

            # Sample depth at object centroid — use median of 5×5 patch for robustness
            x_lo, x_hi = max(0, cx-2), min(w, cx+3)
            y_lo, y_hi = max(0, cy-2), min(h, cy+3)
            depth_z = float(np.median(depth_map[y_lo:y_hi, x_lo:x_hi]))

            # Map pixel coords → normalized world space
            world_x = (cx / w) * 2.0 - 1.0
            world_y = -((cy / h) * 2.0 - 1.0)  # Flip Y

            scene_objects.append({
                "class_id":   class_id,
                "confidence": confidence,
                "bbox":       (x1, y1, x2, y2),
                "depth_z":    depth_z,
                "world_pos":  (world_x, world_y, depth_z),
                "pixel_pos":  (cx, cy),
            })

        return scene_objects
