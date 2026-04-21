"""
ForensicAI Configuration
Central configuration for all application settings.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
MODEL_DIR = BASE_DIR / "data" / "models"
SAMPLE_DIR = BASE_DIR / "data" / "sample_images"
WEB_DIR = BASE_DIR / "web"

# Create directories on import
for d in [UPLOAD_DIR, RESULTS_DIR, MODEL_DIR, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("FORENSICAI_HOST", "127.0.0.1")
PORT = int(os.getenv("FORENSICAI_PORT", 8000))
DEBUG = os.getenv("FORENSICAI_DEBUG", "true").lower() == "true"
MAX_UPLOAD_SIZE_MB = 25  # Maximum file upload size

# ──────────────────────────────────────────────
# ELA Settings
# ──────────────────────────────────────────────
ELA_QUALITY = 90          # JPEG re-compression quality (lower = more sensitive)
ELA_SCALE = 15            # Amplification factor for difference visualization
ELA_THRESHOLD = 40        # Pixel intensity threshold for manipulation flags

# ──────────────────────────────────────────────
# Copy-Move Detection Settings
# ──────────────────────────────────────────────
COPYMOVE_MAX_FEATURES = 10000    # Max ORB keypoints to extract
COPYMOVE_MATCH_THRESHOLD = 0.75  # Distance ratio for good matches
COPYMOVE_MIN_MATCHES = 15        # Minimum matches to flag forgery
COPYMOVE_RANSAC_THRESH = 5.0     # RANSAC reprojection threshold

# ──────────────────────────────────────────────
# Noise Analysis Settings
# ──────────────────────────────────────────────
NOISE_BLOCK_SIZE = 32           # Block size for noise variance analysis
NOISE_WAVELET = "db4"           # Wavelet type for decomposition
NOISE_THRESHOLD_FACTOR = 2.5   # Z-score threshold for anomaly blocks

# ──────────────────────────────────────────────
# AI Detection Settings
# ──────────────────────────────────────────────
AI_MODEL_NAME = "efficientnet_b0"
AI_INPUT_SIZE = 224
AI_CONFIDENCE_THRESHOLD = 0.5
AI_MODEL_PATH = MODEL_DIR / "forensicai_detector.pth"

# ──────────────────────────────────────────────
# Heatmap Settings
# ──────────────────────────────────────────────
HEATMAP_ALPHA = 0.5           # Overlay opacity
HEATMAP_COLORMAP = "jet"      # Colormap for heatmap visualization
HEATMAP_BLUR_KERNEL = 21      # Gaussian blur kernel for smoothing

# ──────────────────────────────────────────────
# Analysis Weights (for overall score)
# ──────────────────────────────────────────────
MODULE_WEIGHTS = {
    "ela": 0.25,
    "copymove": 0.20,
    "noise": 0.15,
    "ai_detection": 0.25,
    "metadata": 0.15,
}
