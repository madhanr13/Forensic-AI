"""
ForensicAI — Module 2: Copy-Move Forgery Detection

Detects when a region of an image has been copied and pasted within the same
image (e.g., cloning/removing objects). Uses ORB keypoint detection and brute-
force matching with geometric consistency filtering via RANSAC.

Technique:
    1. Extract ORB keypoints and descriptors.
    2. Match descriptors with BFMatcher (Hamming distance).
    3. Filter matches by distance ratio (Lowe's ratio test).
    4. Remove self-matches (keypoints matching their own neighborhood).
    5. Verify geometric consistency with RANSAC.
    6. Flag image if sufficient geometrically consistent matches remain.
"""

import numpy as np
import cv2

from forensics.base import BaseAnalyzer, AnalysisResult
from app.config import (
    COPYMOVE_MAX_FEATURES,
    COPYMOVE_MATCH_THRESHOLD,
    COPYMOVE_MIN_MATCHES,
    COPYMOVE_RANSAC_THRESH,
)
from utils.image_utils import image_to_base64


class CopyMoveDetector(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "copymove"

    @property
    def display_name(self) -> str:
        return "Copy-Move Forgery Detection"

    @property
    def description(self) -> str:
        return "Detects cloned (copied & pasted) regions within the same image using feature matching."

    def analyze(self, image: np.ndarray, image_path=None) -> AnalysisResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ── Step 1: ORB Keypoint Extraction ─────────────────────────────
        orb = cv2.ORB_create(nfeatures=COPYMOVE_MAX_FEATURES)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return self._no_detection_result()

        # ── Step 2: Self-matching with BFMatcher ────────────────────────
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=5)

        # ── Step 3: Filter — ratio test + exclude self-matches ──────────
        good_matches = []
        for match_group in matches:
            for m in match_group:
                # Skip self-match (same index)
                if m.queryIdx == m.trainIdx:
                    continue

                # Spatial distance filter — ignore matches too close
                pt1 = np.array(keypoints[m.queryIdx].pt)
                pt2 = np.array(keypoints[m.trainIdx].pt)
                spatial_dist = np.linalg.norm(pt1 - pt2)

                if spatial_dist < 30:  # Points too close, likely same region
                    continue

                if m.distance < COPYMOVE_MATCH_THRESHOLD * 256:
                    good_matches.append(m)
                    break  # Only best non-self match per query

        # Deduplicate match pairs
        seen = set()
        unique_matches = []
        for m in good_matches:
            pair = tuple(sorted([m.queryIdx, m.trainIdx]))
            if pair not in seen:
                seen.add(pair)
                unique_matches.append(m)
        good_matches = unique_matches

        # ── Step 4: Geometric consistency with RANSAC ───────────────────
        forgery_detected = False
        inlier_count = 0
        src_pts = None
        dst_pts = None

        if len(good_matches) >= COPYMOVE_MIN_MATCHES:
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, COPYMOVE_RANSAC_THRESH)

            if mask is not None:
                inlier_count = int(mask.sum())
                if inlier_count >= COPYMOVE_MIN_MATCHES:
                    forgery_detected = True
                    # Keep only inliers
                    inlier_mask = mask.ravel().astype(bool)
                    src_pts = src_pts[inlier_mask].reshape(-1, 2)
                    dst_pts = dst_pts[inlier_mask].reshape(-1, 2)

        # ── Step 5: Score & Visualization ───────────────────────────────
        if forgery_detected:
            score = min(1.0, inlier_count / (COPYMOVE_MIN_MATCHES * 3))
        else:
            score = min(0.3, len(good_matches) / (COPYMOVE_MIN_MATCHES * 2))

        # Visualization
        vis = image.copy()
        if forgery_detected and src_pts is not None:
            for (sx, sy), (dx, dy) in zip(src_pts.astype(int), dst_pts.astype(int)):
                cv2.circle(vis, (sx, sy), 4, (255, 50, 50), -1)
                cv2.circle(vis, (dx, dy), 4, (50, 50, 255), -1)
                cv2.line(vis, (sx, sy), (dx, dy), (0, 255, 0), 1, cv2.LINE_AA)

            # Draw convex hulls around matched regions
            if len(src_pts) >= 3:
                hull_src = cv2.convexHull(src_pts.astype(np.int32))
                hull_dst = cv2.convexHull(dst_pts.astype(np.int32))
                cv2.drawContours(vis, [hull_src], 0, (0, 255, 255), 2)
                cv2.drawContours(vis, [hull_dst], 0, (0, 255, 255), 2)

        # ── Flags ───────────────────────────────────────────────────────
        flags = []
        if forgery_detected:
            flags.append(f"Copy-move forgery detected with {inlier_count} geometrically consistent matches")
            flags.append("Source and destination regions highlighted in the visualization")
        elif len(good_matches) > COPYMOVE_MIN_MATCHES // 2:
            flags.append(f"{len(good_matches)} similar feature matches found (below threshold)")

        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=score,
            confidence=0.8 if forgery_detected else 0.5,
            verdict=self.score_to_verdict(score),
            details={
                "total_keypoints": len(keypoints),
                "initial_matches": len(good_matches),
                "ransac_inliers": inlier_count,
                "forgery_detected": forgery_detected,
                "min_matches_threshold": COPYMOVE_MIN_MATCHES,
            },
            visualization_b64=image_to_base64(vis),
            flags=flags,
        )

    def _no_detection_result(self) -> AnalysisResult:
        return AnalysisResult(
            module_name=self.name,
            display_name=self.display_name,
            description=self.description,
            score=0.0,
            confidence=0.3,
            verdict="Authentic",
            details={
                "total_keypoints": 0,
                "initial_matches": 0,
                "ransac_inliers": 0,
                "forgery_detected": False,
                "note": "Insufficient keypoints for analysis",
            },
            flags=["Insufficient features for copy-move analysis"],
        )
