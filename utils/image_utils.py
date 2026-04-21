"""
ForensicAI — Image Utility Functions
Common image loading, resizing, and encoding helpers.
"""

import base64
import io
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image


def load_image(source, max_dim: int = 2048) -> np.ndarray:
    """
    Load an image from a file path or bytes, returning a BGR numpy array.
    Large images are resized to keep the longest edge ≤ max_dim.
    """
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot read image: {source}")
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes")
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img


def image_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """Encode a BGR numpy image to a base64 string."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> np.ndarray:
    """Decode a base64 string back to a BGR numpy image."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize img to match target_shape (h, w)."""
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def ensure_3channel(img: np.ndarray) -> np.ndarray:
    """Ensure image has 3 channels."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
