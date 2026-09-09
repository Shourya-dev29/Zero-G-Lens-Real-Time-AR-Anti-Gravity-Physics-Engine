"""
depth_estimator.py: MiDaS v2.1 Small pre/post-processing for Zero-G Lens.

MiDaS produces relative inverse depth (disparity): larger values = closer to camera.
Output is normalized to [0, 1] and resized back to the original frame dimensions.
"""
import numpy as np
import cv2
from typing import Tuple

# MiDaS v2.1 Small normalization constants (ImageNet mean/std)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_midas(frame: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize and normalize BGR frame for MiDaS v2.1 Small.

    Output: [1, 3, size, size] float32 tensor (ImageNet-normalized RGB).
    """
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized  = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    normed   = (resized - _MEAN) / _STD                   # Normalize per channel
    tensor   = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
    return tensor


def postprocess_midas(
    depth_raw: np.ndarray,
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Post-process raw MiDaS output into a normalized depth map.

    Args:
        depth_raw:   Raw model output, shape [1, H, W] or [H, W], float32 inverse depth.
        orig_shape:  (height, width) of the original webcam frame for upscaling.

    Returns:
        (orig_h, orig_w) float32 depth map, values normalized to [0, 1].
        Value of 1.0 = closest to camera; 0.0 = farthest.
    """
    depth = depth_raw.squeeze()  # Remove batch dim → (H, W)

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Upscale to original frame resolution
    orig_h, orig_w = orig_shape
    depth_upscaled = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return depth_upscaled.astype(np.float32)


def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    """Convert normalized depth map to a BGR colormap image (for debug visualization)."""
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
