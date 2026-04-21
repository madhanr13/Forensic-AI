"""
ForensicAI — Module 3: Noise Pattern Analysis

Detects image manipulation by analyzing noise consistency across the image.
Authentic images have uniform sensor noise; composited images have regions with
different noise characteristics because they originate from different sources.

Technique:
    1. Extract high-frequency noise residual using wavelet decomposition.
    2. Divide the noise residual into blocks.
    3. Compute statistical features per block (variance, kurtosis).
    4. Flag blocks that deviate significantly from the image-wide distribution.
"""

import numpy as np
import cv2
import pywt
from scipy import stats as scipy_stats

from forensics.base import BaseAnalyzer, AnalysisResult
from app.config import NOISE_BLOCK_SIZE, NOISE_WAVELET, NOISE_THRESHOLD_FACTOR
from utils.image_utils import image_to_base64
from utils.visualization import create_heatmap, overlay_heatmap


class NoiseAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "noise"

    @property
    def display_name(self) -> str:
        return "Noise Pattern Analysis"

    @property
    def description(self) -> str:
        return "Detects composited regions by analyzing inconsistencies in sensor noise patterns."

    def analyze(self, image: np.ndarray, image_path=None) -> AnalysisResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # ── Step 1: Extract noise residual via wavelet decomposition ────
        coeffs = pywt.dwt2(gray, NOISE_WAVELET)
        _, (cH, cV, cD) = coeffs  # Detail coefficients = high-frequency (noise)

        # Combine detail coefficients for overall noise estimate
        noise_residual = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)

        # ── Step 2: Block-wise statistical analysis ─────────────────────
        nr_h, nr_w = noise_residual.shape
        bs = NOISE_BLOCK_SIZE
        blocks_h = nr_h // bs
        blocks_w = nr_w // bs

        if blocks_h == 0 or blocks_w == 0:
            return self._small_image_result()

        block_variances = np.zeros((blocks_h, blocks_w))
        block_kurtoses = np.zeros((blocks_h, blocks_w))

        for by in range(blocks_h):
            for bx in range(blocks_w):
                block = noise_residual[
                    by * bs : (by + 1) * bs,
                    bx * bs : (bx + 1) * bs,
                ]
                block_variances[by, bx] = np.var(block)
                flat = block.flatten()
                if flat.std() > 1e-8:
                    block_kurtoses[by, bx] = scipy_stats.kurtosis(flat, fisher=True)
                else:
                    block_kurtoses[by, bx] = 0.0

        # ── Step 3: Anomaly detection ───────────────────────────────────
        var_mean = block_variances.mean()
        var_std = block_variances.std()

        # Z-score for each block
        if var_std > 1e-8:
            z_scores = np.abs(block_variances - var_mean) / var_std
        else:
            z_scores = np.zeros_like(block_variances)

        anomaly_mask = z_scores > NOISE_THRESHOLD_FACTOR
        anomaly_ratio = anomaly_mask.sum() / anomaly_mask.size if anomaly_mask.size > 0 else 0

        # Coefficient of variation of noise variance
        noise_cv = var_std / (var_mean + 1e-8)

        # ── Step 4: Score ───────────────────────────────────────────────
        score = min(1.0, anomaly_ratio * 0.5 + noise_cv * 0.5)

        # ── Step 5: Visualization — noise inconsistency heatmap ─────────
        # Normalize z-scores to [0, 1] for heatmap
        z_norm = np.clip(z_scores / (NOISE_THRESHOLD_FACTOR * 2), 0, 1)
        heatmap = create_heatmap(z_norm, (h, w))
        vis = overlay_heatmap(image, heatmap, alpha=0.4)

        # ── Flags ───────────────────────────────────────────────────────
        flags = []
        if anomaly_ratio > 0.05:
            pct = anomaly_ratio * 100
            flags.append(f"{pct:.1f}% of blocks show anomalous noise patterns")
        if noise_cv > 1.0:
            flags.append("High variation in noise distribution — possible compositing")

        # Identify regions with highest anomaly
        if anomaly_mask.any():
            max_z_idx = np.unravel_index(z_scores.argmax(), z_scores.shape)
            flags.append(
                f"Strongest anomaly at block ({max_z_idx[1]}, {max_z_idx[0]}) "
                f"with z-score {z_scores[max_z_idx]:.2f}"
            )

        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=score,
            confidence=min(1.0, 0.4 + anomaly_ratio * 2),
            verdict=self.score_to_verdict(score),
            details={
                "mean_noise_variance": round(float(var_mean), 4),
                "noise_std": round(float(var_std), 4),
                "noise_cv": round(float(noise_cv), 4),
                "anomalous_block_ratio": round(float(anomaly_ratio), 4),
                "total_blocks": int(anomaly_mask.size),
                "anomalous_blocks": int(anomaly_mask.sum()),
                "wavelet": NOISE_WAVELET,
                "block_size": NOISE_BLOCK_SIZE,
            },
            visualization_b64=image_to_base64(vis),
            flags=flags,
        )

    def _small_image_result(self) -> AnalysisResult:
        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=0.0,
            confidence=0.1,
            verdict="Authentic",
            details={"note": "Image too small for block-wise noise analysis"},
            flags=["Image too small for reliable noise analysis"],
        )
