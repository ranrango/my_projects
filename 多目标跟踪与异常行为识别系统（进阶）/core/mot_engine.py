# core/mot_engine.py
import cv2
import numpy as np
from .detector import YOLOv8Detector
from .bytetracker import BYTETracker

class MOTEngine:
    def __init__(
        self,
        onnx_model,
        track_thresh=0.3,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        detector_config=None,
        verbose=False,
    ):
        detector_config = detector_config or {}
        self.detector = YOLOv8Detector(onnx_model, verbose=verbose, **detector_config)
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
        self.frame_id = 0
        self.verbose = verbose

    def process_frame(self, frame):
        self.frame_id += 1
        boxes, scores, class_ids = self.detector.detect(frame)
        
        if self.verbose:
            print(f"  Detector: {len(boxes)} boxes, classes: {class_ids}")
        
        # 构建检测输入: [x1, y1, x2, y2, score]
        dets = []
        valid_class_ids = []  # 存储有效的类别ID
        
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            # 确保坐标有效
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append([x1, y1, x2, y2, score, cls_id])
            valid_class_ids.append(cls_id)
        
        if len(dets) == 0:
            if self.verbose:
                print("  No valid detections")
            return []
        
        dets = np.array(dets, dtype=np.float32)
        img_h, img_w = frame.shape[:2]
        if self.verbose:
            print(f"  Sending to tracker: {dets.shape[0]} detections")
        
        online_targets = self.tracker.update(dets, (img_h, img_w), (img_h, img_w))
        if self.verbose:
            print(f"  Tracker returned: {len(online_targets)} targets")
        
        # 关键：将类别ID映射到跟踪结果
        tracked_objects = []
        for i, t in enumerate(online_targets):
            if i >= len(valid_class_ids):
                continue
            x1, y1, x2, y2 = t.tlbr
            # 确保坐标顺序正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            tracked_objects.append({
                'id': t.track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'score': t.score,
                'class_id': t.class_id if t.class_id is not None and t.class_id >= 0 else valid_class_ids[i]
            })
        return tracked_objects
