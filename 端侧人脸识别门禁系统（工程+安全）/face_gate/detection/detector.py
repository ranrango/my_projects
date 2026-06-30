from __future__ import annotations

import cv2
import numpy as np


class FaceDetector:
    """Haar-cascade based face detector, zero external ML dependencies."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_face_size=(80, 80)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = tuple(min_face_size)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(
                "Haar cascade not found. Make sure opencv-python is properly installed."
            )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return list of face dicts with keys: bbox (x1,y1,x2,y2), confidence."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(rects) == 0:
            return []
        faces = []
        for (x, y, w, h) in rects:
            faces.append(
                {
                    "bbox": (int(x), int(y), int(x + w), int(y + h)),
                    "confidence": 1.0,
                }
            )
        return faces

    def crop_face(self, frame: np.ndarray, bbox: tuple, margin: float = 0.2) -> np.ndarray:
        """Crop and resize a face region with optional margin."""
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        dx, dy = int(w * margin), int(h * margin)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w_img, x2 + dx)
        y2 = min(h_img, y2 + dy)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112))
