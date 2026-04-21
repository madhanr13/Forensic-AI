"""
ForensicAI — Module 1: Error Level Analysis (ELA)

Detects image manipulation by re-compressing the image at a known JPEG quality
and amplifying the difference. Tampered/spliced regions appear brighter because
they were saved at a different compression level than the surrounding area.

Technique:
    1. Re-save image as JPEG at a fixed quality (e.g. 90%).
    2. Compute absolute pixel difference between original and re-compressed.
    3. Amplify the difference by a scale factor.
    4. Analyze block-wise deviation to compute a manipulation score.
"""

import io
import numpy as np
import cv2
from PIL import Image

from forensics.base import BaseAnalyzer, AnalysisResult
from app.config import ELA_QUALITY, ELA_SCALE, ELA_THRESHOLD
from utils.image_utils import image_to_base64
from utils.visualization import normalize_array


class ELAAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "ela"

    @property
    def display_name(self) -> str:
        return "Error Level Analysis"

    @property
    def description(self) -> str:
        return "Detects spliced or pasted regions by analyzing JPEG compression artifacts."

    def analyze(self, image: np.ndarray, image_path=None) -> AnalysisResult:
        # ── Step 1: Re-compress at known quality ────────────────────────
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = np.array(Image.open(buf))
        recompressed = cv2.cvtColor(recompressed, cv2.COLOR_RGB2BGR)

        # Ensure shapes match (resize if pixel-level mismatch from encoding)
        if recompressed.shape != image.shape:
            recompressed = cv2.resize(recompressed, (image.shape[1], image.shape[0]))

        # ── Step 2: Compute & amplify difference ────────────────────────
        diff = cv2.absdiff(image, recompressed)
        ela_image = np.clip(diff.astype(np.float64) * ELA_SCALE, 0, 255).astype(np.uint8)

        # ── Step 3: Block-wise analysis ─────────────────────────────────
        gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
        block_size = 16
        h, w = gray_ela.shape
        blocks_h, blocks_w = h // block_size, w // block_size

        block_means = np.zeros((blocks_h, blocks_w))
        for by in range(blocks_h):
            for bx in range(blocks_w):
                block = gray_ela[
                    by * block_size : (by + 1) * block_size,
                    bx * block_size : (bx + 1) * block_size,
                ]
                block_means[by, bx] = block.mean()

        # ── Step 4: Score computation ───────────────────────────────────
        overall_mean = block_means.mean()
        overall_std = block_means.std()

        # High-intensity blocks are suspicious
        high_blocks = (block_means > ELA_THRESHOLD).sum()
        total_blocks = blocks_h * blocks_w
        high_ratio = high_blocks / total_blocks if total_blocks > 0 else 0

        # Variation across blocks indicates inconsistent compression
        cv_score = (overall_std / (overall_mean + 1e-8))  # coefficient of variation

        # Combine into a manipulation score (0–1)
        score = min(1.0, high_ratio * 0.6 + cv_score * 0.4)

        # ── Step 5: Create visualization ────────────────────────────────
        # Enhance the ELA image for visibility
        ela_display = cv2.normalize(ela_image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a colormap for a more informative visualization
        ela_gray = cv2.cvtColor(ela_display, cv2.COLOR_BGR2GRAY)
        ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

        # ── Build flags ─────────────────────────────────────────────────
        flags = []
        if high_ratio > 0.1:
            flags.append(f"{high_blocks} blocks ({high_ratio:.0%}) show high error levels")
        if cv_score > 0.8:
            flags.append("High variation in compression artifacts across image")
        if overall_mean > ELA_THRESHOLD * 0.8:
            flags.append("Elevated overall error level suggests re-encoding or editing")

        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=score,
            confidence=min(1.0, 0.5 + abs(score - 0.5)),
            verdict=self.score_to_verdict(score),
            details={
                "overall_mean_error": round(float(overall_mean), 2),
                "overall_std": round(float(overall_std), 2),
                "high_error_block_ratio": round(float(high_ratio), 4),
                "coefficient_of_variation": round(float(cv_score), 4),
                "quality_used": ELA_QUALITY,
                "scale_factor": ELA_SCALE,
                "total_blocks_analyzed": int(total_blocks),
            },
            visualization_b64=image_to_base64(ela_colored),
            flags=flags,
        )
