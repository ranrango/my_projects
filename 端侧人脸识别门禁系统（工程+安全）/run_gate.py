"""
run_gate.py  —  生产运行入口

用法:
  python run_gate.py                              # 使用默认摄像头
  python run_gate.py --source test.mp4            # 视频文件
  python run_gate.py --no-display                 # 无界面（服务器部署）
  python run_gate.py --config config/custom.json  # 自定义配置
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_gate.detection.detector import FaceDetector
from face_gate.detection.liveness import LivenessChecker
from face_gate.recognition.gallery import GalleryIndex, Recognizer
from face_gate.security.guard import SecurityGuard
from face_gate.security.access_controller import AccessController
from face_gate.utils.audit_logger import AuditLogger
from face_gate.utils.config import load_config
from face_gate.utils.visualizer import draw_face, draw_gate_status, draw_fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Edge Face-Gate access control system.")
    p.add_argument("--config", default="config/default_config.json")
    p.add_argument("--source", default="", help="Video path, RTSP URL, or camera index.")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(args.config, base_dir=base_dir)

    if args.source:
        cfg["camera"]["source"] = int(args.source) if args.source.isdigit() else args.source
    if args.no_display:
        cfg["display"] = False

    index_path = cfg["enrollment"]["index_file"]
    if not os.path.exists(index_path):
        print(f"[WARN] Gallery index not found at {index_path}.")
        print("       Run enroll.py first to register faces.")

    detector = FaceDetector(
        scale_factor=cfg["detection"]["scale_factor"],
        min_neighbors=cfg["detection"]["min_neighbors"],
        min_face_size=cfg["detection"]["min_face_size"],
    )
    liveness = LivenessChecker(
        texture_var_threshold=cfg["liveness"]["texture_var_threshold"]
    )
    recognizer = Recognizer(
        match_threshold=cfg["recognition"]["match_threshold"],
        unknown_label=cfg["recognition"]["unknown_label"],
    )
    recognizer.load_gallery(index_path)

    security = SecurityGuard(
        max_failed=cfg["security"]["max_failed_attempts"],
        lockout_sec=cfg["security"]["lockout_duration_sec"],
    )
    controller = AccessController(
        allow_unknown=cfg["access"]["allow_unknown"],
        gate_open_duration=cfg["access"]["gate_open_duration_sec"],
        cooldown_sec=cfg["access"]["cooldown_sec"],
        unknown_label=cfg["recognition"]["unknown_label"],
    )
    audit = AuditLogger(
        log_file=cfg["audit"]["log_file"],
        snapshot_dir=cfg["audit"]["snapshot_dir"],
    )

    source = cfg["camera"]["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    print(f"[INFO] Face-Gate started  source={source}  gallery={len(recognizer.gallery)} faces")
    frame_id = 0
    t0 = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video source ended.")
            break
        frame_id += 1

        if frame_id % 10 == 0:
            elapsed = time.time() - t0
            fps_display = frame_id / max(elapsed, 1e-6)

        faces = detector.detect(frame)
        for face in faces:
            bbox = face["bbox"]
            crop = detector.crop_face(frame, bbox)

            decision = "DENY"
            identity = cfg["recognition"]["unknown_label"]
            score = 0.0

            if cfg["liveness"]["enabled"]:
                is_live, liveness_score = liveness.check(crop)
                if not is_live:
                    decision = "LIVENESS_FAIL"
                    if args.verbose:
                        print(f"[LIVENESS] frame={frame_id} score={liveness_score:.1f} FAIL")
                    audit.log("liveness_fail", identity, frame_id, frame,
                              liveness_score=liveness_score)
                    draw_face(frame, bbox, "SPOOF?", liveness_score, decision)
                    continue

            identity, score = recognizer.identify(crop)

            if security.is_locked(identity):
                remaining = security.lockout_remaining(identity)
                decision = "LOCKED"
                if args.verbose:
                    print(f"[LOCKED] {identity} locked for {remaining:.0f}s more")
                audit.log("locked", identity, frame_id, frame, score=score)
                draw_face(frame, bbox, identity, score, decision)
                continue

            decision_result = controller.decide(identity, score)
            if decision_result == AccessController.GRANT:
                decision = "GRANT"
                controller.open_gate(identity, cfg["access"]["gate_open_duration_sec"])
                security.record_success(identity)
                print(f"[GRANT] frame={frame_id} identity={identity} score={score:.3f}")
                audit.log("grant", identity, frame_id, frame, score=score)
            else:
                decision = "DENY"
                triggered_lockout = security.record_failure(identity)
                print(f"[DENY]  frame={frame_id} identity={identity} score={score:.3f}")
                audit.log("deny", identity, frame_id, frame, score=score,
                          lockout_triggered=triggered_lockout)

            draw_face(frame, bbox, identity, score, decision)

        draw_gate_status(frame, controller.gate_is_open)
        draw_fps(frame, fps_display)

        if cfg.get("display", True):
            cv2.imshow("Face-Gate  [q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if cfg.get("log_interval") and frame_id % cfg["log_interval"] == 0:
            print(f"[RUN] frame={frame_id} fps={fps_display:.1f} gate={'OPEN' if controller.gate_is_open else 'closed'}")

        if args.max_frames and frame_id >= args.max_frames:
            print(f"[INFO] Reached max-frames={args.max_frames}")
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"[INFO] Done. {frame_id} frames in {elapsed:.1f}s ({frame_id/max(elapsed,1e-6):.1f} fps)")


if __name__ == "__main__":
    main()
