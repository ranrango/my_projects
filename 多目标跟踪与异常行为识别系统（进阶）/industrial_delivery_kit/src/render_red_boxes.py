import argparse
import os
import sys
import time


KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(KIT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import cv2

from core.mot_engine import MOTEngine


CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def draw_red_boxes(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        class_id = obj.get("class_id", -1)
        track_id = obj.get("id", -1)
        score = obj.get("score", 0.0)
        label = f"{CLASS_NAMES.get(class_id, 'obj')} ID:{track_id} {score:.2f}"

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        text_y = max(24, y1 - 8)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Render detected people/vehicles with red boxes.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mot = MOTEngine(
        args.model,
        track_thresh=args.conf,
        frame_rate=int(fps) if fps > 1 else 30,
        detector_config={
            "input_size": 640,
            "conf_thres": args.conf,
            "iou_thres": 0.45,
            "target_class_ids": [0, 1, 2, 3, 5, 7],
        },
    )

    frame_id = 0
    started = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracked = mot.process_frame(frame)
        frame = draw_red_boxes(frame, tracked)
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = max(time.time() - started, 1e-6)
    print(f"red_box_video={args.output}")
    print(f"frames={frame_id}")
    print(f"runtime_fps={frame_id / elapsed:.3f}")


if __name__ == "__main__":
    main()
