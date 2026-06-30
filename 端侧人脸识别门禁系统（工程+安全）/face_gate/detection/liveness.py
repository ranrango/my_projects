from __future__ import annotations

import cv2
import numpy as np


class LivenessChecker:
    """Texture-based anti-spoofing: rejects flat printed photos.

    Two complementary checks are run:
    1. Laplacian variance — printed photos are blurry / low-frequency.
    2. Edge density — real faces have richer fine detail.

    Neither method is foolproof against 3-D masks or HD screens.
    For a production system, replace or combine with a CNN-based liveness model.
    """

    def __init__(self, texture_var_threshold: float = 200.0):
        self.texture_var_threshold = texture_var_threshold

    def check(self, face_crop: np.ndarray) -> tuple[bool, float]:
        """Returns (is_live, score).  score >= threshold → live."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()
        return score >= self.texture_var_threshold, float(score)
