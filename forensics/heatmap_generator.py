"""
ForensicAI — Module 5: Tampering Heatmap Generator

Produces a unified tampering probability heatmap by fusing signals
from multiple forensic modules (ELA, noise analysis) and deep
feature analysis.  The fused heatmap highlights regions most likely
to have been manipulated.

Technique:
    1. Compute ELA residual map.
    2. Compute noise inconsistency map (wavelet-based).
    3. Compute edge density anomaly map.
    4. Normalize each map to [0, 1] and combine with learned /
       configurable weights.
    5. Smooth & render as a colour heatmap overlay.
"""

import numpy as np
import cv2
import pywt

from forensics.base import BaseAnalyzer, AnalysisResult
from utils.image_utils import image_to_base64
from utils.visualization import create_heatmap, overlay_heatmap, normalize_array
from app.config import (
    ELA_QUALITY,
    ELA_SCALE,
    NOISE_BLOCK_SIZE,
    NOISE_WAVELET,
    HEATMAP_ALPHA,
    HEATMAP_BLUR_KERNEL,
)

from PIL import Image
import io


class HeatmapGenerator(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "heatmap"

    @property
    def display_name(self) -> str:
        return "Tampering Localization Heatmap"

    @property
    def description(self) -> str:
        return "Fuses multiple forensic signals into a unified heatmap showing likely tampered regions."

    def analyze(self, image: np.ndarray, image_path=None) -> AnalysisResult:
        h, w = image.shape[:2]

        # ── Signal 1: ELA-based ─────────────────────────────────────────
        ela_map = self._compute_ela_map(image)

        # ── Signal 2: Noise variance map ────────────────────────────────
        noise_map = self._compute_noise_map(image)

        # ── Signal 3: Edge density anomaly ──────────────────────────────
        edge_map = self._compute_edge_anomaly_map(image)

        # ── Resize all maps to same shape ───────────────────────────────
        target = (h, w)
        ela_map = cv2.resize(ela_map, (w, h))
        noise_map = cv2.resize(noise_map, (w, h))
        edge_map = cv2.resize(edge_map, (w, h))

        # Normalize each to [0, 1]
        ela_map = normalize_array(ela_map)
        noise_map = normalize_array(noise_map)
        edge_map = normalize_array(edge_map)

        # ── Fuse signals ────────────────────────────────────────────────
        weights = {"ela": 0.45, "noise": 0.35, "edge": 0.20}
        fused = (
            weights["ela"] * ela_map
            + weights["noise"] * noise_map
            + weights["edge"] * edge_map
        )
        fused = normalize_array(fused)

        # ── Create visualization ────────────────────────────────────────
        fused_uint8 = (fused * 255).astype(np.uint8)
        fused_smooth = cv2.GaussianBlur(fused_uint8, (HEATMAP_BLUR_KERNEL, HEATMAP_BLUR_KERNEL), 0)
        heatmap = cv2.applyColorMap(fused_smooth, cv2.COLORMAP_JET)
        vis = overlay_heatmap(image, heatmap, alpha=HEATMAP_ALPHA)

        # ── Score: overall manipulation likelihood ──────────────────────
        # Use the top 5% of pixels as indicator
        top_5pct = np.percentile(fused, 95)
        top_1pct = np.percentile(fused, 99)
        avg_score = fused.mean()

        score = min(1.0, top_5pct * 0.5 + top_1pct * 0.3 + avg_score * 0.2)

        # ── Flags ───────────────────────────────────────────────────────
        flags = []
        if top_5pct > 0.6:
            flags.append("Strong manipulation signals detected in localized regions")
        if top_1pct > 0.8:
            flags.append("Very high confidence tampering in specific areas")
        if avg_score > 0.4:
            flags.append("Elevated manipulation indicators across broader image regions")

        # Find hotspot location
        max_loc = np.unravel_index(fused_smooth.argmax(), fused_smooth.shape)
        flags.append(f"Primary hotspot at pixel ({max_loc[1]}, {max_loc[0]})")

        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=score,
            confidence=min(1.0, 0.5 + score * 0.5),
            verdict=self.score_to_verdict(score),
            details={
                "top_5_percentile": round(float(top_5pct), 4),
                "top_1_percentile": round(float(top_1pct), 4),
                "average_score": round(float(avg_score), 4),
                "fusion_weights": weights,
                "hotspot_x": int(max_loc[1]),
                "hotspot_y": int(max_loc[0]),
            },
            visualization_b64=image_to_base64(vis),
            flags=flags,
        )

    def _compute_ela_map(self, image: np.ndarray) -> np.ndarray:
        """Compute per-pixel ELA residual."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recomp = np.array(Image.open(buf))
        recomp = cv2.cvtColor(recomp, cv2.COLOR_RGB2BGR)
        if recomp.shape != image.shape:
            recomp = cv2.resize(recomp, (image.shape[1], image.shape[0]))
        diff = cv2.absdiff(image, recomp).astype(np.float64)
        ela = np.mean(diff, axis=2)  # Average across channels
        return ela

    def _compute_noise_map(self, image: np.ndarray) -> np.ndarray:
        """Compute block-wise noise variance map."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        coeffs = pywt.dwt2(gray, NOISE_WAVELET)
        _, (cH, cV, cD) = coeffs
        noise = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)

        bs = NOISE_BLOCK_SIZE
        h, w = noise.shape
        blocks_h, blocks_w = max(1, h // bs), max(1, w // bs)
        var_map = np.zeros((blocks_h, blocks_w))

        for by in range(blocks_h):
            for bx in range(blocks_w):
                block = noise[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs]
                var_map[by, bx] = np.var(block)

        # Deviation from mean
        mean_var = var_map.mean()
        std_var = var_map.std() + 1e-8
        deviation_map = np.abs(var_map - mean_var) / std_var
        return deviation_map

    def _compute_edge_anomaly_map(self, image: np.ndarray) -> np.ndarray:
        """Compute block-wise edge density anomaly map."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float64)

        bs = 32
        h, w = edges.shape
        blocks_h, blocks_w = max(1, h // bs), max(1, w // bs)
        density_map = np.zeros((blocks_h, blocks_w))

        for by in range(blocks_h):
            for bx in range(blocks_w):
                block = edges[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs]
                density_map[by, bx] = block.mean()

        # Deviation from mean
        m = density_map.mean()
        s = density_map.std() + 1e-8
        anomaly = np.abs(density_map - m) / s
        return anomaly
