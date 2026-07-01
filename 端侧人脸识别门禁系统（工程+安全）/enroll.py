"""
enroll.py  —  人脸注册 CLI

功能:
  1. 从目录批量注册: python enroll.py build  --faces-dir enrolled_faces
  2. 摄像头采集并注册: python enroll.py capture --identity person_001 --count 10
  3. 列出已注册人员:    python enroll.py list

目录结构约定（适用于 build 模式）:
  enrolled_faces/
    person_001/
      001.jpg
      002.jpg
    person_002/
      001.jpg
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_gate.detection.detector import FaceDetector
from face_gate.recognition.gallery import build_gallery_from_dir, GalleryIndex
from face_gate.recognition.embedder import FaceEmbedder
from face_gate.utils.config import load_config


def cmd_build(args, cfg):
    faces_dir = args.faces_dir or cfg["enrollment"]["faces_dir"]
    index_path = args.index or cfg["enrollment"]["index_file"]
    min_images = args.min_images or cfg["enrollment"]["min_enrollment_images"]
    print(f"Building gallery from: {faces_dir}")
    build_gallery_from_dir(faces_dir, index_path, min_images=min_images)


def cmd_capture(args, cfg):
    identity = args.identity
    count = args.count
    faces_dir = args.faces_dir or cfg["enrollment"]["faces_dir"]
    out_dir = os.path.join(faces_dir, identity)
    os.makedirs(out_dir, exist_ok=True)

    detector = FaceDetector()
    source = args.source if args.source else cfg["camera"]["source"]
    if isinstance(source, str) and str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    saved = 0
    print(f"Capturing {count} frames for '{identity}'. Press SPACE to capture, Q to quit.")
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect(frame)
        display = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {saved}/{count}  SPACE=capture  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and faces:
            path = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"  Saved {path}")
            time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {saved} images for '{identity}' in {out_dir}")

    if saved >= cfg["enrollment"]["min_enrollment_images"]:
        print("Re-building gallery index...")
        index_path = args.index or cfg["enrollment"]["index_file"]
        build_gallery_from_dir(
            cfg["enrollment"]["faces_dir"],
            index_path,
            min_images=cfg["enrollment"]["min_enrollment_images"],
        )
    else:
        print(f"[WARN] Only {saved} images saved (need {cfg['enrollment']['min_enrollment_images']}). Gallery not updated.")


def cmd_list(args, cfg):
    index_path = args.index or cfg["enrollment"]["index_file"]
    gallery = GalleryIndex.load(index_path)
    if len(gallery) == 0:
        print("Gallery is empty or index not found.")
        return
    from collections import Counter
    counts = Counter(gallery.labels)
    print(f"Gallery: {len(gallery)} embeddings, {len(counts)} identities")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n} embedding(s)")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Face enrollment CLI")
    p.add_argument("--config", default="config/default_config.json")
    sub = p.add_subparsers(dest="cmd")

    build_p = sub.add_parser("build", help="Build gallery from a directory of face images")
    build_p.add_argument("--faces-dir", default="")
    build_p.add_argument("--index", default="")
    build_p.add_argument("--min-images", type=int, default=0)

    cap_p = sub.add_parser("capture", help="Capture face images from camera and enroll")
    cap_p.add_argument("--identity", required=True)
    cap_p.add_argument("--count", type=int, default=10)
    cap_p.add_argument("--faces-dir", default="")
    cap_p.add_argument("--index", default="")
    cap_p.add_argument("--source", default="")

    list_p = sub.add_parser("list", help="List enrolled identities")
    list_p.add_argument("--index", default="")

    args = p.parse_args()
    cfg = load_config(args.config, base_dir=base_dir)

    if args.cmd == "build":
        cmd_build(args, cfg)
    elif args.cmd == "capture":
        cmd_capture(args, cfg)
    elif args.cmd == "list":
        cmd_list(args, cfg)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
