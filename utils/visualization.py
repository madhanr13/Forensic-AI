"""
ForensicAI — Visualization Utilities
Heatmap generation, overlay blending, and color-mapping helpers.
"""

import cv2
import numpy as np
from typing import Tuple


def create_heatmap(
    scores: np.ndarray,
    shape: Tuple[int, int],
    colormap: int = cv2.COLORMAP_JET,
    blur_kernel: int = 21,
) -> np.ndarray:
    """
    Generate a smooth heatmap from a 2D score array.

    Args:
        scores: 2D array of values in [0, 1].
        shape: Target (height, width) for the heatmap.
        colormap: OpenCV colormap constant.
        blur_kernel: Gaussian blur kernel size for smoothing.

    Returns:
        BGR heatmap image (uint8).
    """
    # Normalize to 0–255
    normalized = np.clip(scores * 255, 0, 255).astype(np.uint8)

    # Resize to target shape
    resized = cv2.resize(normalized, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    # Smooth
    if blur_kernel > 0:
        resized = cv2.GaussianBlur(resized, (blur_kernel, blur_kernel), 0)

    # Apply colormap
    heatmap = cv2.applyColorMap(resized, colormap)
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a heatmap over the original image."""
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def draw_matches_on_image(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    circle_radius: int = 5,
) -> np.ndarray:
    """Draw lines between matched keypoint pairs on an image."""
    result = image.copy()
    for (sx, sy), (dx, dy) in zip(src_points.astype(int), dst_points.astype(int)):
        cv2.circle(result, (sx, sy), circle_radius, (255, 0, 0), -1)
        cv2.circle(result, (dx, dy), circle_radius, (0, 0, 255), -1)
        cv2.line(result, (sx, sy), (dx, dy), color, thickness)
    return result


def create_comparison_image(
    original: np.ndarray,
    processed: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Analysis",
) -> np.ndarray:
    """Create a side-by-side comparison image."""
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]

    # Match heights
    target_h = max(h1, h2)
    if h1 != target_h:
        scale = target_h / h1
        original = cv2.resize(original, None, fx=scale, fy=scale)
    if h2 != target_h:
        scale = target_h / h2
        processed = cv2.resize(processed, None, fx=scale, fy=scale)

    # Divider
    divider = np.ones((target_h, 3, 3), dtype=np.uint8) * 100

    # Concatenate
    combined = np.hstack([original, divider, processed])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label_left, (10, 30), font, 0.8, (0, 255, 255), 2)
    cv2.putText(combined, label_right, (original.shape[1] + 15, 30), font, 0.8, (0, 255, 255), 2)

    return combined


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - mn) / (mx - mn)
