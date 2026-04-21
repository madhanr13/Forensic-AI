"""
ForensicAI — Base Analyzer
Abstract base class that all forensic analysis modules must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class AnalysisResult:
    """Standardized result from any forensic analysis module."""
    module_name: str
    display_name: str
    description: str
    score: float                       # 0.0 (authentic) → 1.0 (manipulated)
    confidence: float                  # 0.0 → 1.0 confidence in the result
    verdict: str                       # "Authentic", "Suspicious", "Manipulated"
    details: dict = field(default_factory=dict)
    visualization_b64: Optional[str] = None   # Base64-encoded result image
    flags: list = field(default_factory=list)  # Specific red flags found

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "display_name": self.display_name,
            "description": self.description,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
            "details": self.details,
            "visualization": self.visualization_b64,
            "flags": self.flags,
        }


class BaseAnalyzer(ABC):
    """Abstract base class for all forensic analysis modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier for this module."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the dashboard."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this module detects."""
        ...

    @abstractmethod
    def analyze(self, image: np.ndarray, image_path: Optional[str] = None) -> AnalysisResult:
        """
        Run the forensic analysis on a single image.

        Args:
            image: Input image as a NumPy array (BGR, uint8).
            image_path: Optional original file path (needed for metadata analysis).

        Returns:
            AnalysisResult with score, verdict, visualization, and flags.
        """
        ...

    @staticmethod
    def score_to_verdict(score: float) -> str:
        """Convert a 0–1 manipulation score to a human-readable verdict."""
        if score < 0.3:
            return "Authentic"
        elif score < 0.6:
            return "Suspicious"
        else:
            return "Manipulated"
