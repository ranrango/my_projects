import datetime
import json
import os


class AuditLogger:
    def __init__(self, log_file: str, snapshot_dir: str):
        self.log_file = log_file
        self.snapshot_dir = snapshot_dir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

    def _entry(self, event_type: str, identity: str, frame_id: int, meta: dict) -> dict:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "identity": identity,
            "frame_id": frame_id,
            **meta,
        }

    def log(self, event_type: str, identity: str, frame_id: int, frame=None, **meta):
        entry = self._entry(event_type, identity, frame_id, meta)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if frame is not None:
            self._save_snapshot(frame, frame_id, event_type, identity)

        return entry

    def _save_snapshot(self, frame, frame_id: int, event_type: str, identity: str):
        import cv2
        safe_id = "".join(c if c.isalnum() else "_" for c in str(identity))
        fname = f"{frame_id:08d}_{event_type}_{safe_id}.jpg"
        path = os.path.join(self.snapshot_dir, fname)
        cv2.imwrite(path, frame)
        return path
