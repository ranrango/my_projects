import os
import tempfile

import cv2
import numpy as np

from anomaly.anomaly_engine import AnomalyEngine
from core.bytetracker import BYTETracker
from core.mot_engine import MOTEngine
from utils.config import load_config
from utils.event_writer import EventWriter
from utils.roi import RoiFilter


def validate_tracker():
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, frame_rate=30)
    frames = [
        np.array([[100, 100, 200, 300, 0.90, 0]], dtype=np.float32),
        np.array([[104, 102, 204, 302, 0.88, 0]], dtype=np.float32),
        np.array([[110, 106, 210, 306, 0.86, 0]], dtype=np.float32),
    ]
    ids = []
    for dets in frames:
        tracks = tracker.update(dets, (720, 1280), (720, 1280))
        ids.append([t.track_id for t in tracks])
    assert ids == [[1], [1], [1]], ids


def validate_anomaly_and_events():
    engine = AnomalyEngine(
        fps=1,
        crowd_threshold=3,
        crowd_radius=100,
        crowd_duration=0,
        loitering_duration=1,
        alert_cooldown=1,
    )
    objects = [
        {"id": 1, "bbox": [10, 10, 50, 100], "center": (30, 55), "class_id": 0},
        {"id": 2, "bbox": [40, 10, 80, 100], "center": (60, 55), "class_id": 0},
        {"id": 3, "bbox": [70, 10, 110, 100], "center": (90, 55), "class_id": 0},
    ]
    engine.update(objects, 1)
    alerts = engine.update(objects, 2)
    assert any(alert["type"] == "crowd" for alert in alerts), alerts

    with tempfile.TemporaryDirectory() as tmpdir:
        event_path = os.path.join(tmpdir, "events.jsonl")
        snapshot_dir = os.path.join(tmpdir, "snapshots")
        writer = EventWriter(event_path, snapshot_dir, "test_camera")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        events = writer.write_alerts(alerts, frame, 1, 0.033)
        assert events
        assert os.path.exists(event_path)
        assert os.path.exists(events[0]["snapshot_path"])


def validate_roi():
    roi = RoiFilter(enabled=True, polygon=[[0, 0], [100, 0], [100, 100], [0, 100]])
    objects = [
        {"id": 1, "center": (50, 50)},
        {"id": 2, "center": (150, 50)},
    ]
    assert [obj["id"] for obj in roi.filter_objects(objects)] == [1]


def validate_detection_smoke(base_dir):
    model_path = os.path.join(base_dir, "models", "yolov8n.onnx")
    video_path = os.path.join(base_dir, "test.mp4")
    if not (os.path.exists(model_path) and os.path.exists(video_path)):
        return
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Unable to read test video"
    mot = MOTEngine(
        model_path,
        track_thresh=0.5,
        frame_rate=30,
        detector_config={
            "input_size": 640,
            "conf_thres": 0.5,
            "iou_thres": 0.45,
            "target_class_ids": [0, 2, 5, 7],
        },
    )
    tracked = mot.process_frame(frame)
    assert isinstance(tracked, list)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, "config", "default_config.json"), base_dir)
    assert os.path.exists(config["model_path"])
    validate_tracker()
    validate_anomaly_and_events()
    validate_roi()
    validate_detection_smoke(base_dir)
    print("✅ system validation passed")


if __name__ == "__main__":
    main()
