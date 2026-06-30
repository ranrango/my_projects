import argparse
import json
import os
import sys
import time


KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(KIT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import cv2

from anomaly.anomaly_engine import AnomalyEngine
from core.mot_engine import MOTEngine
from utils.event_writer import EventWriter
from utils.roi import RoiFilter
from utils.visualizer import draw_alerts, draw_tracks


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(config_path, value):
    if value in ("", None):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str) and "://" in value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(config_path), value))


def build_anomaly(config, fps):
    cfg = config["anomaly"]
    return AnomalyEngine(
        fps=fps,
        person_class_id=cfg["person_class_id"],
        crowd_enabled=cfg["crowd_enabled"],
        crowd_threshold=cfg["crowd_threshold"],
        crowd_radius=cfg["crowd_radius"],
        crowd_duration=cfg["crowd_duration_sec"],
        fall_enabled=cfg["fall_enabled"],
        fall_speed_thresh=cfg["fall_speed_thresh"],
        fall_ratio_thresh=cfg["fall_ratio_thresh"],
        loitering_enabled=cfg["loitering_enabled"],
        loitering_area=cfg["loitering_area"],
        loitering_duration=cfg["loitering_duration_sec"],
        alert_cooldown=config["events"]["cooldown_sec"],
    )


def main():
    parser = argparse.ArgumentParser(description="Industrial edge runtime reference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_json(config_path)
    source = resolve_path(config_path, args.source or config["source"])
    model_path = resolve_path(config_path, args.model or config["model_path"])
    output_dir = resolve_path(config_path, config["output_dir"])
    max_frames = args.max_frames or config.get("max_frames", 0)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    if isinstance(source, str) and "://" not in source and not os.path.exists(source):
        raise FileNotFoundError(f"source not found: {source}")

    os.makedirs(output_dir, exist_ok=True)
    event_path = os.path.join(output_dir, "industrial_events.jsonl")
    snapshot_dir = os.path.join(output_dir, "snapshots")
    video_path = os.path.join(output_dir, "industrial_output.mp4")
    if os.path.exists(event_path):
        os.remove(event_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config["tracker"]["frame_rate"] or 30
    fps = int(fps) if fps > 1 else 30
    tracker_cfg = config["tracker"]
    mot = MOTEngine(
        model_path,
        track_thresh=tracker_cfg["track_thresh"],
        track_buffer=tracker_cfg["track_buffer"],
        match_thresh=tracker_cfg["match_thresh"],
        frame_rate=fps,
        detector_config=config["detector"],
        verbose=args.verbose,
    )
    anomaly = build_anomaly(config, fps)
    roi = RoiFilter(**config["roi"])
    writer = EventWriter(event_path, snapshot_dir, config["camera_id"])
    video_writer = None

    frame_id = 0
    event_count = 0
    started = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracks = roi.filter_objects(mot.process_frame(frame))
        alerts = anomaly.update(tracks, frame_id) if config["anomaly"]["enabled"] else []
        frame = draw_tracks(frame, tracks)
        frame = draw_alerts(frame, alerts)
        frame = roi.draw(frame)
        event_count += len(writer.write_alerts(alerts, frame, frame_id, frame_id / fps))

        if config["events"]["save_video"]:
            if video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            video_writer.write(frame)

        if config.get("display", False):
            cv2.imshow("industrial edge runtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and frame_id >= max_frames:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if config.get("display", False):
        cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 1e-6)
    summary = {
        "site_id": config["site_id"],
        "camera_id": config["camera_id"],
        "frames": frame_id,
        "events": event_count,
        "runtime_fps": round(frame_id / elapsed, 3),
        "event_path": event_path,
        "snapshot_dir": snapshot_dir,
        "video_path": video_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
