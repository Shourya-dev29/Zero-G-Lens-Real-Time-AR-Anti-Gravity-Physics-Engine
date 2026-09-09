"""
ar_overlay.py: AR Rendering Pipeline for Zero-G Lens.

Composites physics simulation results onto live webcam frames using OpenCV.
All rendering runs on CPU — the GPU (Adreno) is deliberately kept free for
potential future Vulkan-based rendering or display compositing.

Rendering layers (bottom to top):
  1. Raw webcam frame (background)
  2. Depth edge overlay (optional debug mode)
  3. Motion trail polylines per physics body
  4. Physics body glow circles (size ∝ depth, alpha blended)
  5. Velocity vector arrows
  6. Bounding box outlines + object labels
  7. Hand landmarks and gesture labels
  8. HUD (performance telemetry)
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional

from pipeline.object_detector import get_label


class ARRenderer:
    # ── Palette (BGR) ───────────────────────────────────────────────
    PALETTE = {
        "bbox":        (0,   255, 120),   # Neon green
        "trail":       (80,  160, 255),   # Electric blue
        "hand_lm":     (255,  60,  60),   # Coral red
        "gesture_lbl": (255, 200,  50),   # Amber
        "velocity":    (255, 190,   0),   # Yellow
        "glow":        (0,   220, 140),   # Cyan-green
        "text":        (255, 255, 255),   # White
        "hud_bg":      ( 15,  15,  15),   # Near-black
    }
    # High-contrast overrides (WCAG AA)
    HC_PALETTE = {
        "bbox":        (255, 255,   0),
        "trail":       (255, 255, 255),
        "hand_lm":     (255,   0, 255),
        "gesture_lbl": (255, 255,   0),
        "velocity":    (  0, 255, 255),
        "glow":        (255, 255,   0),
        "text":        (255, 255, 255),
        "hud_bg":      (  0,   0,   0),
    }

    TRAIL_LEN   = 35   # Maximum trail length (frames)
    FONT        = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, high_contrast: bool = False):
        self.high_contrast = high_contrast
        self.colors        = self.HC_PALETTE if high_contrast else self.PALETTE
        self.trails        = defaultdict(lambda: deque(maxlen=self.TRAIL_LEN))
        cv2.namedWindow("Zero-G Lens", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Zero-G Lens", cv2.WND_PROP_TOPMOST, 1)

    # ── World → Pixel Coordinate Transform ─────────────────────────
    @staticmethod
    def world_to_pixel(world_pos: tuple, frame_shape: tuple) -> tuple:
        """Map normalized world coords [-1,1] to pixel coordinates."""
        h, w = frame_shape[:2]
        px = int((world_pos[0] + 1.0) / 2.0 * w)
        py = int((1.0 - (world_pos[1] + 1.0) / 2.0) * h)
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        return (px, py)

    # ── Main Render Method ──────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        physics_state: Dict[int, dict],
        hand_data: Dict[str, Any],
        detections: Optional[list] = None,
    ) -> np.ndarray:
        """
        Composite all AR layers onto a copy of the webcam frame.
        Returns the composited output frame (BGR uint8).
        """
        out = frame.copy()

        # ── Layer 1: Bounding boxes (if detections provided) ────────
        if detections:
            self._draw_bboxes(out, detections)

        # ── Layers 2–4: Physics overlays ────────────────────────────
        for cid, state in physics_state.items():
            pos = state["pos"]
            vel = state["vel"]
            px, py = self.world_to_pixel(pos, out.shape)

            # Update motion trail
            self.trails[cid].append((px, py))
            self._draw_trail(out, cid)

            # Glow circle (depth-proportional radius)
            depth_val = float(pos[2])
            radius    = int(np.clip(25 + depth_val * 30, 12, 60))
            self._draw_glow(out, (px, py), radius, self.colors["glow"])

            # Velocity vector arrow
            speed = float(np.linalg.norm(vel))
            if speed > 0.02:
                scale_factor = min(40.0 / speed, 25.0)  # Cap arrow length
                ax = int(np.clip(px + vel[0] * scale_factor, 0, out.shape[1]-1))
                ay = int(np.clip(py - vel[1] * scale_factor, 0, out.shape[0]-1))
                cv2.arrowedLine(out, (px, py), (ax, ay),
                                self.colors["velocity"], 2, tipLength=0.35)

            # Class label (if class ID maps to COCO label)
            label = get_label(cid)
            cv2.putText(out, label, (px + 8, py - 8),
                        self.FONT, 0.55, self.colors["text"], 1, cv2.LINE_AA)

        # ── Layer 5: Hand landmarks and gesture labels ───────────────
        if hand_data and hand_data.get("hands"):
            self._draw_hands(out, hand_data["hands"])

        return out

    def _draw_bboxes(self, frame: np.ndarray, detections: list):
        for class_id, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["bbox"], 2)
            label = f"{get_label(class_id)} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, self.FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-lh-6), (x1+lw+4, y1), self.colors["bbox"], -1)
            cv2.putText(frame, label, (x1+2, y1-4),
                        self.FONT, 0.5, self.colors["hud_bg"], 1, cv2.LINE_AA)

    def _draw_trail(self, frame: np.ndarray, cid: int):
        trail = list(self.trails[cid])
        if len(trail) < 2:
            return
        for i in range(1, len(trail)):
            alpha  = i / len(trail)
            color  = tuple(int(c * alpha) for c in self.colors["trail"])
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

    def _draw_glow(self, frame: np.ndarray, center: tuple, radius: int, color: tuple):
        """Alpha-blended filled circle for glow effect."""
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)

    def _draw_hands(self, frame: np.ndarray, hands: list):
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),       # Thumb
            (0,5),(5,6),(6,7),(7,8),       # Index
            (0,9),(9,10),(10,11),(11,12),  # Middle
            (0,13),(13,14),(14,15),(15,16),# Ring
            (0,17),(17,18),(18,19),(19,20),# Pinky
            (5,9),(9,13),(13,17),          # Palm
        ]
        for hand in hands:
            lm_px   = hand.get("landmarks_px", [])
            gesture = hand.get("gesture", "")
            wrist   = hand.get("wrist_px", (0, 0))

            # Draw skeleton connections
            for a, b in CONNECTIONS:
                if a < len(lm_px) and b < len(lm_px):
                    cv2.line(frame, lm_px[a], lm_px[b],
                             self.colors["hand_lm"], 1, cv2.LINE_AA)

            # Draw landmark points
            for pt in lm_px:
                cv2.circle(frame, pt, 4, self.colors["hand_lm"], -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

            # Gesture label above wrist
            if gesture and gesture not in ("UNKNOWN", ""):
                lbl_pt = (wrist[0] - 40, wrist[1] - 20)
                lbl_pt = (max(0, lbl_pt[0]), max(20, lbl_pt[1]))
                cv2.putText(frame, gesture, lbl_pt,
                            self.FONT, 0.8, self.colors["gesture_lbl"], 2, cv2.LINE_AA)

    # ── HUD Overlay ─────────────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray, stats: dict):
        """Render performance telemetry HUD in the top-left corner."""
        lines = [
            f"Latency  {stats.get('latency_ms', 0):>6.1f} ms",
            f"FPS      {stats.get('fps', 0):>6.1f}",
            f"P95 lat  {stats.get('p95_ms', 0):>6.1f} ms",
            f"Objects  {stats.get('num_objects', 0):>6d}",
            f"NPU      {'ON' if stats.get('npu_active', True) else 'CPU':>6s}",
        ]
        bg_w, bg_h = 200, len(lines) * 22 + 10
        overlay    = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + bg_w, 5 + bg_h),
                      self.colors["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        for i, line in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(frame, line, (12, y),
                        self.FONT, 0.50, self.colors["text"], 1, cv2.LINE_AA)

    # ── Display ─────────────────────────────────────────────────────
    def show(self, frame: np.ndarray):
        cv2.imshow("Zero-G Lens", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            raise SystemExit("User quit.")
        if key == ord("h"):
            # Toggle high-contrast mode at runtime
            self.high_contrast = not self.high_contrast
            self.colors = self.HC_PALETTE if self.high_contrast else self.PALETTE
