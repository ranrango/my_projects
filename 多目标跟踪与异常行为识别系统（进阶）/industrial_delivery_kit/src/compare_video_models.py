import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict


KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(KIT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import cv2

from core.detector import YOLOv8Detector


CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def summarize(counter):
    return {CLASS_NAMES.get(int(k), str(k)): int(v) for k, v in sorted(counter.items())}


def main():
    parser = argparse.ArgumentParser(description="Compare two YOLO class filters on the same video.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    previous = YOLOv8Detector(
        args.model,
        conf_thres=args.conf,
        iou_thres=0.45,
        target_class_ids=[0, 2, 5, 7],
    )
    current = YOLOv8Detector(
        args.model,
        conf_thres=args.conf,
        iou_thres=0.45,
        target_class_ids=[0, 1, 2, 3, 5, 7],
    )

    previous_counts = Counter()
    current_counts = Counter()
    per_frame_rows = []
    frame_id = 0
    started = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        _, _, prev_classes = previous.detect(frame)
        _, _, curr_classes = current.detect(frame)

        prev_counter = Counter(prev_classes)
        curr_counter = Counter(curr_classes)
        previous_counts.update(prev_counter)
        current_counts.update(curr_counter)
        per_frame_rows.append({
            "frame_id": frame_id,
            "previous_total": sum(prev_counter.values()),
            "current_total": sum(curr_counter.values()),
            "previous_by_class": json.dumps(summarize(prev_counter), ensure_ascii=False),
            "current_by_class": json.dumps(summarize(curr_counter), ensure_ascii=False),
        })

        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release()
    elapsed = max(time.time() - started, 1e-6)
    previous_total = sum(previous_counts.values())
    current_total = sum(current_counts.values())
    extra_counts = current_counts.copy()
    extra_counts.subtract(previous_counts)
    extra_counts = Counter({k: v for k, v in extra_counts.items() if v > 0})

    summary = {
        "source": os.path.abspath(args.source),
        "model": os.path.abspath(args.model),
        "video": {
            "frames_in_file": total_frames,
            "frames_processed": frame_id,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
        },
        "previous_model": {
            "name": "previous_person_vehicle",
            "target_classes": ["person", "car", "bus", "truck"],
            "total_detections": previous_total,
            "by_class": summarize(previous_counts),
        },
        "current_model": {
            "name": "current_red_box_extended_vehicle",
            "target_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
            "total_detections": current_total,
            "by_class": summarize(current_counts),
        },
        "difference": {
            "additional_detections": current_total - previous_total,
            "additional_by_class": summarize(extra_counts),
        },
        "runtime": {
            "seconds": round(elapsed, 3),
            "fps": round(frame_id / elapsed, 3),
        },
    }

    summary_path = os.path.join(args.output_dir, "model_compare_summary.json")
    csv_path = os.path.join(args.output_dir, "model_compare_per_frame.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "previous_total",
                "current_total",
                "previous_by_class",
                "current_by_class",
            ],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"csv_path={csv_path}")


if __name__ == "__main__":
    main()
