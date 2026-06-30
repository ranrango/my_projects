import cv2
import numpy as np


class RoiFilter:
    def __init__(self, enabled=False, polygon=None):
        self.enabled = bool(enabled)
        self.polygon = polygon or []
        self._points = None
        if self.enabled and len(self.polygon) >= 3:
            self._points = np.array(self.polygon, dtype=np.int32)
        else:
            self.enabled = False

    def contains(self, point):
        if not self.enabled:
            return True
        return cv2.pointPolygonTest(self._points, point, False) >= 0

    def filter_objects(self, tracked_objects):
        return [obj for obj in tracked_objects if self.contains(obj["center"])]

    def draw(self, frame):
        if self.enabled:
            cv2.polylines(frame, [self._points], isClosed=True, color=(0, 255, 255), thickness=2)
        return frame
