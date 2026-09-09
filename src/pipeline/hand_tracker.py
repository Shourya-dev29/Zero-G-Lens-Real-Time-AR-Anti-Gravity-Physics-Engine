"""
hand_tracker.py: MediaPipe Hand landmark detection pre/post-processing for Zero-G Lens.

The QNN-compiled MediaPipe Hand model outputs 21 3D hand landmarks per hand.
Post-processing classifies gestures from landmark geometry using rule-based heuristics:
  - PINCH:     Distance between index tip (lm 8) and thumb tip (lm 4) < threshold
  - OPEN_PALM: All 4 non-thumb fingers extended (tip above MCP joint in image space)
  - FIST:      All fingers curled (tips below PIP joints)
  - POINT:     Only index finger extended
  - PEACE:     Index and middle fingers extended, ring + pinky curled

Landmark indices follow MediaPipe Hand Landmark topology:
  https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models

Landmark index reference (key ones):
  0  = Wrist
  4  = Thumb tip
  8  = Index tip        12 = Middle tip
  16 = Ring tip         20 = Pinky tip
  5  = Index MCP        9  = Middle MCP
  13 = Ring MCP        17  = Pinky MCP
  6  = Index PIP        10 = Middle PIP
  14 = Ring PIP        18  = Pinky PIP
"""
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any


def preprocess_hand(frame: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize and normalize BGR frame for MediaPipe Hand model.
    Output: [1, 3, size, size] float32 tensor, RGB, normalized to [0, 1].
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor  = resized.astype(np.float32) / 255.0
    tensor  = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # [1,3,H,W]
    return tensor


def postprocess_hand(
    raw_output: List[np.ndarray],
    frame_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Parse raw MediaPipe Hand model output into structured landmark data.

    Args:
        raw_output:  List of model output tensors.
                     Assumes raw_output[0] shape [1, 63] = 21 landmarks × (x, y, z)
                     Values in [0, 1] normalized to input image size.
        frame_shape: (height, width) of original frame for pixel coord conversion.

    Returns:
        {
          "hands": [
            {
              "landmarks_norm": [(x, y, z), ...],   # 21 normalized [0,1] coords
              "landmarks_px":   [(px, py), ...],     # pixel coords in original frame
              "wrist_pos_normalized": (x, y),        # wrist [0,1]
              "wrist_px": (px, py),
              "gesture": str,                         # classified gesture name
            },
            ...
          ]
        }
    """
    oh, ow = frame_shape
    result = {"hands": []}

    if not raw_output or raw_output[0] is None:
        return result

    landmarks_flat = raw_output[0].squeeze()  # [63] or [N, 63]

    # Handle single vs multi-hand output
    if landmarks_flat.ndim == 1:
        all_hands = [landmarks_flat]
    elif landmarks_flat.ndim == 2:
        all_hands = [landmarks_flat[i] for i in range(landmarks_flat.shape[0])]
    else:
        return result

    for hand_lms in all_hands:
        if len(hand_lms) < 63:
            continue

        # Reshape [63] → [21, 3]
        lm = hand_lms.reshape(21, 3)

        landmarks_norm = [(float(lm[i, 0]), float(lm[i, 1]), float(lm[i, 2]))
                          for i in range(21)]
        landmarks_px   = [(int(np.clip(lm[i, 0] * ow, 0, ow-1)),
                           int(np.clip(lm[i, 1] * oh, 0, oh-1)))
                          for i in range(21)]

        wrist_norm = (float(lm[0, 0]), float(lm[0, 1]))
        wrist_px   = landmarks_px[0]
        gesture    = _classify_gesture(lm)

        result["hands"].append({
            "landmarks_norm":        landmarks_norm,
            "landmarks_px":          landmarks_px,
            "wrist_pos_normalized":  wrist_norm,
            "wrist_px":              wrist_px,
            "gesture":               gesture,
        })

    return result


def _classify_gesture(lm: np.ndarray) -> str:
    """
    Rule-based gesture classifier using MediaPipe hand landmark geometry.

    Args:
        lm: [21, 3] array of (x, y, z) normalized landmarks.
            x: left-right [0,1], y: top-bottom [0,1] (image convention).

    Note: In image space, y increases downward.
          "Finger extended" = tip.y < PIP.y (tip is ABOVE PIP joint in image).
    """
    # Tip and joint landmark indices
    TIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
    PIPS = [3, 7, 11, 15, 19]   # Corresponding PIP / IP joints

    # ── Pinch: index tip close to thumb tip ─────────────────────────
    thumb_tip = lm[4, :2]
    index_tip = lm[8, :2]
    pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
    if pinch_dist < 0.07:
        return "PINCH"

    # ── Finger extension flags ──────────────────────────────────────
    # For thumb: compare x-axis (thumb extends horizontally)
    thumb_extended = bool(lm[4, 0] < lm[3, 0])  # Works for right hand
    # For other fingers: tip.y < pip.y → extended upward
    index_ext  = bool(lm[8,  1] < lm[7,  1])
    middle_ext = bool(lm[12, 1] < lm[11, 1])
    ring_ext   = bool(lm[16, 1] < lm[15, 1])
    pinky_ext  = bool(lm[20, 1] < lm[19, 1])

    fingers_up = [index_ext, middle_ext, ring_ext, pinky_ext]
    n_extended = sum(fingers_up)

    # ── Gesture rules ───────────────────────────────────────────────
    if n_extended == 0:
        return "FIST"
    if n_extended == 4:
        return "OPEN_PALM"
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "POINT"
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "PEACE"

    return "UNKNOWN"
