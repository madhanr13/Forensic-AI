"""
ForensicAI — Analysis Orchestrator

Coordinates all forensic analysis modules, aggregates their results,
and computes an overall authenticity score using configurable weights.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensics.ela_analyzer import ELAAnalyzer
from forensics.copymove_detector import CopyMoveDetector
from forensics.noise_analyzer import NoiseAnalyzer
from forensics.ai_detector import AIDetector
from forensics.heatmap_generator import HeatmapGenerator
from forensics.metadata_analyzer import MetadataAnalyzer
from forensics.base import BaseAnalyzer, AnalysisResult
from utils.image_utils import load_image
from app.config import MODULE_WEIGHTS


class ForensicOrchestrator:
    """
    Coordinates all forensic analysis modules.

    Usage:
        orchestrator = ForensicOrchestrator()
        results = orchestrator.analyze_all("path/to/image.jpg")
    """

    def __init__(self):
        self.modules: Dict[str, BaseAnalyzer] = {
            "ela": ELAAnalyzer(),
            "copymove": CopyMoveDetector(),
            "noise": NoiseAnalyzer(),
            "ai_detection": AIDetector(),
            "heatmap": HeatmapGenerator(),
            "metadata": MetadataAnalyzer(),
        }

    def analyze_all(self, image_path: str) -> dict:
        """
        Run all forensic modules and return aggregated results.

        Returns dict with:
            - overall_score: Weighted combination of module scores (0–100)
            - overall_verdict: "Authentic" / "Suspicious" / "Manipulated"
            - modules: Dict of individual module results
        """
        image = load_image(image_path)
        module_results = {}

        for name, analyzer in self.modules.items():
            try:
                result = analyzer.analyze(image, image_path=image_path)
                module_results[name] = result.to_dict()
            except Exception as e:
                module_results[name] = {
                    "module_name": name,
                    "display_name": analyzer.display_name,
                    "description": analyzer.description,
                    "score": 0.0,
                    "confidence": 0.0,
                    "verdict": "Error",
                    "details": {"error": str(e)},
                    "visualization": None,
                    "flags": [f"Module error: {str(e)}"],
                }

        # ── Compute overall score ───────────────────────────────────────
        weighted_score = 0.0
        total_weight = 0.0

        for module_name, weight in MODULE_WEIGHTS.items():
            if module_name in module_results:
                result = module_results[module_name]
                if result["verdict"] != "Error":
                    weighted_score += result["score"] * weight * result["confidence"]
                    total_weight += weight * result["confidence"]

        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0

        # Convert to 0–100 scale
        overall_score_100 = round(overall_score * 100, 1)

        # Overall verdict
        if overall_score < 0.3:
            overall_verdict = "Authentic"
        elif overall_score < 0.6:
            overall_verdict = "Suspicious"
        else:
            overall_verdict = "Manipulated"

        return {
            "overall_score": overall_score_100,
            "overall_verdict": overall_verdict,
            "modules": module_results,
        }

    def analyze_single(self, image_path: str, module_name: str) -> dict:
        """Run a single forensic module."""
        if module_name not in self.modules:
            raise ValueError(
                f"Unknown module: {module_name}. "
                f"Available: {list(self.modules.keys())}"
            )

        image = load_image(image_path)
        analyzer = self.modules[module_name]
        result = analyzer.analyze(image, image_path=image_path)
        return result.to_dict()
