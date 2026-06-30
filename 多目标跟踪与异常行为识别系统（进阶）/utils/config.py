import copy
import json
import os


DEFAULT_CONFIG = {
    "camera_id": "demo_camera_001",
    "model_path": "models/yolov8n.onnx",
    "source": "test.mp4",
    "output_video": "runs/demo_output.mp4",
    "event_jsonl": "runs/events.jsonl",
    "snapshot_dir": "runs/snapshots",
    "display": False,
    "save_video": True,
    "max_frames": 0,
    "log_interval": 30,
    "reconnect_attempts": 3,
    "reconnect_delay_sec": 1.0,
    "detector": {
        "input_size": 640,
        "conf_thres": 0.5,
        "iou_thres": 0.45,
        "target_class_ids": [0, 2, 5, 7],
    },
    "tracker": {
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "frame_rate": 30,
    },
    "roi": {
        "enabled": False,
        "polygon": [],
    },
    "anomaly": {
        "enabled": True,
        "person_class_id": 0,
        "crowd_enabled": True,
        "crowd_threshold": 3,
        "crowd_radius": 100,
        "crowd_duration_sec": 2.0,
        "fall_enabled": True,
        "fall_speed_thresh": 15,
        "fall_ratio_thresh": 0.7,
        "loitering_enabled": True,
        "loitering_area": 50,
        "loitering_duration_sec": 5.0,
        "alert_cooldown_sec": 5.0,
    },
}


def deep_update(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(base_dir, value):
    if value in ("", None):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str) and not os.path.isabs(value) and "://" not in value:
        return os.path.join(base_dir, value)
    return value


def load_config(config_path=None, base_dir=None):
    base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        config = deep_update(config, loaded)

    for key in ("model_path", "source", "output_video", "event_jsonl", "snapshot_dir"):
        config[key] = resolve_path(base_dir, config.get(key))
    return config
