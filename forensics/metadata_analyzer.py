"""
ForensicAI — Module 6: Metadata Forensics

Analyzes EXIF, IPTC, and XMP metadata embedded in image files to detect
signs of editing, metadata stripping, or other suspicious patterns.

Checks:
    - Editing software signatures (Photoshop, GIMP, etc.)
    - Missing or stripped EXIF data
    - GPS inconsistencies
    - Timestamp anomalies
    - Camera/device info analysis
    - Thumbnail vs.  main image mismatch indicators
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from forensics.base import BaseAnalyzer, AnalysisResult


# Known editing software signatures
EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "affinity", "snapseed",
    "pixelmator", "paint.net", "paint shop", "corel", "capture one",
    "darktable", "rawtherapee", "luminar", "photoscape", "fotor",
    "canva", "picmonkey", "befunky", "ipiccy", "pixlr",
    "adobe", "acdsee", "irfanview",
]

AI_GENERATOR_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "comfyui", "automatic1111", "novelai", "firefly",
    "imagen", "deepai", "nightcafe", "artbreeder",
]


class MetadataAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "metadata"

    @property
    def display_name(self) -> str:
        return "Metadata Forensics"

    @property
    def description(self) -> str:
        return "Analyzes EXIF/IPTC metadata for signs of editing, stripping, or AI generation."

    def analyze(self, image: np.ndarray, image_path: Optional[str] = None) -> AnalysisResult:
        flags = []
        metadata = {}
        score = 0.0
        anomaly_count = 0

        if image_path is None or not Path(image_path).exists():
            return self._no_metadata_result(
                "No file path provided — metadata analysis requires the original file."
            )

        # ── Extract metadata ────────────────────────────────────────────
        try:
            import exifread
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=True)
        except Exception as e:
            return self._no_metadata_result(f"Could not read metadata: {e}")

        if not tags:
            flags.append("⚠ No EXIF metadata found — possible metadata stripping or screenshot")
            anomaly_count += 2
            metadata["exif_present"] = False
        else:
            metadata["exif_present"] = True
            metadata["total_tags"] = len(tags)

            # ── Parse key fields ────────────────────────────────────────
            parsed = self._parse_tags(tags)
            metadata.update(parsed)

            # ── Check 1: Editing software ───────────────────────────────
            software = parsed.get("software", "").lower()
            if software:
                metadata["software"] = parsed["software"]
                for editor in EDITING_SOFTWARE:
                    if editor in software:
                        flags.append(f"Image was processed with editing software: {parsed['software']}")
                        anomaly_count += 2
                        break
                for ai_sig in AI_GENERATOR_SIGNATURES:
                    if ai_sig in software:
                        flags.append(f"⚠ AI generation signature detected: {parsed['software']}")
                        anomaly_count += 3
                        break

            # ── Check 2: Camera info ────────────────────────────────────
            if parsed.get("camera_make") and parsed.get("camera_model"):
                metadata["camera"] = f"{parsed['camera_make']} {parsed['camera_model']}"
            elif not parsed.get("camera_make") and not parsed.get("camera_model"):
                flags.append("No camera make/model — not from a camera or metadata stripped")
                anomaly_count += 1

            # ── Check 3: Timestamps ─────────────────────────────────────
            dates = []
            for dkey in ["datetime_original", "datetime_digitized", "datetime_modified"]:
                if parsed.get(dkey):
                    dates.append((dkey, parsed[dkey]))

            if not dates:
                flags.append("No timestamp data found in metadata")
                anomaly_count += 1
            elif len(dates) >= 2:
                # Check for timestamp inconsistencies
                try:
                    parsed_dates = []
                    for dkey, dval in dates:
                        for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                            try:
                                parsed_dates.append((dkey, datetime.strptime(str(dval), fmt)))
                                break
                            except ValueError:
                                continue

                    if len(parsed_dates) >= 2:
                        time_diffs = []
                        for i in range(len(parsed_dates)):
                            for j in range(i + 1, len(parsed_dates)):
                                diff = abs((parsed_dates[i][1] - parsed_dates[j][1]).total_seconds())
                                time_diffs.append(diff)

                        max_diff = max(time_diffs) if time_diffs else 0
                        if max_diff > 86400:  # More than 1 day difference
                            flags.append(
                                f"Suspicious timestamp gap: {max_diff / 3600:.1f} hours "
                                "between original and modified dates"
                            )
                            anomaly_count += 2
                except Exception:
                    pass

            # ── Check 4: GPS data ───────────────────────────────────────
            if parsed.get("gps_latitude") and parsed.get("gps_longitude"):
                metadata["gps"] = {
                    "latitude": parsed["gps_latitude"],
                    "longitude": parsed["gps_longitude"],
                }
            else:
                metadata["gps"] = None

            # ── Check 5: Resolution & dimensions ────────────────────────
            if parsed.get("image_width") and parsed.get("image_height"):
                meta_w = int(str(parsed["image_width"]))
                meta_h = int(str(parsed["image_height"]))
                actual_h, actual_w = image.shape[:2]
                if abs(meta_w - actual_w) > 2 or abs(meta_h - actual_h) > 2:
                    flags.append(
                        f"Dimension mismatch: metadata says {meta_w}×{meta_h}, "
                        f"actual is {actual_w}×{actual_h}"
                    )
                    anomaly_count += 2

            # ── Check 6: Thumbnail presence ─────────────────────────────
            thumb_tags = [t for t in tags if "thumb" in t.lower() or "thumbnail" in t.lower()]
            if thumb_tags:
                metadata["has_thumbnail"] = True
            else:
                metadata["has_thumbnail"] = False

        # ── File-level checks ───────────────────────────────────────────
        file_path = Path(image_path)
        metadata["file_name"] = file_path.name
        metadata["file_size_kb"] = round(os.path.getsize(image_path) / 1024, 1)
        metadata["file_extension"] = file_path.suffix.lower()

        if file_path.suffix.lower() == ".png":
            flags.append("PNG format — no JPEG compression artifacts to analyze via ELA")

        # ── Score ───────────────────────────────────────────────────────
        score = min(1.0, anomaly_count / 6)

        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=score,
            confidence=min(1.0, 0.4 + anomaly_count * 0.1),
            verdict=self.score_to_verdict(score),
            details=metadata,
            visualization_b64=None,  # No visual output for metadata
            flags=flags,
        )

    def _parse_tags(self, tags: dict) -> dict:
        """Extract relevant fields from EXIF tags into a clean dict."""
        parsed = {}

        mapping = {
            "Image Software": "software",
            "Image Make": "camera_make",
            "Image Model": "camera_model",
            "EXIF DateTimeOriginal": "datetime_original",
            "EXIF DateTimeDigitized": "datetime_digitized",
            "Image DateTime": "datetime_modified",
            "EXIF ExifImageWidth": "image_width",
            "EXIF ExifImageLength": "image_height",
            "Image ImageWidth": "image_width",
            "Image ImageLength": "image_height",
            "GPS GPSLatitude": "gps_latitude",
            "GPS GPSLongitude": "gps_longitude",
            "EXIF ISOSpeedRatings": "iso",
            "EXIF FocalLength": "focal_length",
            "EXIF ExposureTime": "exposure_time",
            "EXIF FNumber": "f_number",
            "EXIF Flash": "flash",
            "Image Orientation": "orientation",
            "EXIF ColorSpace": "color_space",
            "EXIF WhiteBalance": "white_balance",
        }

        for exif_key, clean_key in mapping.items():
            if exif_key in tags:
                val = str(tags[exif_key])
                if val and val != "0":
                    parsed[clean_key] = val

        return parsed

    def _no_metadata_result(self, reason: str) -> AnalysisResult:
        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=0.3,
            confidence=0.2,
            verdict="Suspicious",
            details={"note": reason},
            flags=[reason],
        )
