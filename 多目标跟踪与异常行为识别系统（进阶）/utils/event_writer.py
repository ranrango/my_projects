import json
import os
import time

import cv2


class EventWriter:
    def __init__(self, event_jsonl, snapshot_dir, camera_id):
        self.event_jsonl = event_jsonl
        self.snapshot_dir = snapshot_dir
        self.camera_id = camera_id
        os.makedirs(os.path.dirname(event_jsonl), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        open(event_jsonl, "a", encoding="utf-8").close()

    def write_alerts(self, alerts, frame, frame_id, timestamp_sec):
        events = []
        for alert in alerts:
            snapshot_path = self._save_snapshot(alert, frame, frame_id)
            event = {
                "camera_id": self.camera_id,
                "event_type": alert["type"],
                "frame_id": frame_id,
                "timestamp_sec": round(float(timestamp_sec), 3),
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": list(alert.get("location", [])),
                "track_id": alert.get("track_id"),
                "count": alert.get("count"),
                "snapshot_path": snapshot_path,
            }
            self._append_jsonl(event)
            events.append(event)
        return events

    def _save_snapshot(self, alert, frame, frame_id):
        filename = f"{frame_id:06d}_{alert['type']}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def _append_jsonl(self, event):
        with open(self.event_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
