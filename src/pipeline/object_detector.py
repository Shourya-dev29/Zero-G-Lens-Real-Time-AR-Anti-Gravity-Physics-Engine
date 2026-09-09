"""
object_detector.py: YOLOv8n pre/post-processing for Zero-G Lens.

Pre-processing:  BGR frame → normalized, letterboxed [1,3,640,640] tensor
Post-processing: Raw [1,84,8400] output → filtered (class_id, conf, bbox_xyxy) list

YOLO class IDs follow MS COCO 80-class taxonomy.
"""
import numpy as np
import cv2
from typing import List, Tuple

# MS COCO 80-class labels
COCO_LABELS = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def preprocess_yolo(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Letterbox-resize frame to (size × size), normalize to [0,1], convert to
    CHW float32 tensor with batch dimension: [1, 3, size, size].
    """
    # Letterbox: preserve aspect ratio, pad with gray (114)
    h, w   = frame.shape[:2]
    scale  = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas   = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    # BGR → RGB, HWC → CHW, normalize
    rgb      = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor   = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # [1,3,H,W]
    return tensor


def postprocess_yolo(
    output: np.ndarray,
    orig_shape: Tuple[int, int],
    conf_thresh: float = 0.45,
    iou_thresh:  float = 0.45,
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """
    Process raw YOLOv8 output [1, 84, 8400] into filtered detections.

    YOLOv8 output format:
      dim 0-3  : cx, cy, w, h (normalized to 640×640 input)
      dim 4-83 : class confidence scores for 80 COCO classes

    Returns: list of (class_id, confidence, (x1, y1, x2, y2)) in original pixel coords.
    """
    pred       = output[0].T  # [8400, 84]
    oh, ow     = orig_shape    # original frame dimensions for scaling back

    boxes     = []
    scores    = []
    class_ids = []

    for row in pred:
        cx, cy, bw, bh = row[:4]
        class_scores   = row[4:]
        class_id       = int(np.argmax(class_scores))
        confidence     = float(class_scores[class_id])

        if confidence < conf_thresh:
            continue

        # Scale from 640×640 YOLO space → original image space
        # (simplified: assumes square letterbox padding is symmetric)
        scale = max(oh, ow) / 640.0
        x1 = int((cx - bw / 2) * scale)
        y1 = int((cy - bh / 2) * scale)
        x2 = int((cx + bw / 2) * scale)
        y2 = int((cy + bh / 2) * scale)

        # Clamp to image bounds
        x1, x2 = np.clip([x1, x2], 0, ow - 1)
        y1, y2 = np.clip([y1, y2], 0, oh - 1)

        boxes.append([x1, y1, x2 - x1, y2 - y1])  # OpenCV NMS wants x,y,w,h
        scores.append(confidence)
        class_ids.append(class_id)

    # Non-Maximum Suppression
    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    results = []
    for i in indices:
        idx  = int(i)
        x, y, bw, bh = boxes[idx]
        results.append((
            class_ids[idx],
            scores[idx],
            (x, y, x + bw, y + bh),
        ))
    return results


def get_label(class_id: int) -> str:
    """Return COCO class label for a given class ID."""
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"class_{class_id}"
