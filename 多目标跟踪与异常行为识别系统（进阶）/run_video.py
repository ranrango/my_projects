# run_video.py
import argparse
import cv2
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mot_engine import MOTEngine
from anomaly.anomaly_engine import AnomalyEngine
from utils.config import load_config
from utils.event_writer import EventWriter
from utils.roi import RoiFilter
from utils.visualizer import draw_tracks, draw_alerts


def open_capture(source):
    return cv2.VideoCapture(source)


def build_anomaly_engine(config, fps):
    anomaly_cfg = config["anomaly"]
    return AnomalyEngine(
        fps=fps,
        person_class_id=anomaly_cfg["person_class_id"],
        crowd_enabled=anomaly_cfg["crowd_enabled"],
        crowd_threshold=anomaly_cfg["crowd_threshold"],
        crowd_radius=anomaly_cfg["crowd_radius"],
        crowd_duration=anomaly_cfg["crowd_duration_sec"],
        fall_enabled=anomaly_cfg["fall_enabled"],
        fall_speed_thresh=anomaly_cfg["fall_speed_thresh"],
        fall_ratio_thresh=anomaly_cfg["fall_ratio_thresh"],
        loitering_enabled=anomaly_cfg["loitering_enabled"],
        loitering_area=anomaly_cfg["loitering_area"],
        loitering_duration=anomaly_cfg["loitering_duration_sec"],
        alert_cooldown=anomaly_cfg["alert_cooldown_sec"],
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run deployable MOT and anomaly detection.")
    parser.add_argument("--config", default=os.path.join(base_dir, "config", "default_config.json"))
    parser.add_argument("--model", default="", help="Override model path.")
    parser.add_argument("--source", default="", help="Override input source. Use a video path, RTSP URL, or camera index.")
    parser.add_argument("--video", default="", help="Compatibility alias for --source.")
    parser.add_argument("--camera", action="store_true", help="Use camera instead of --video.")
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means full video.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow.")
    parser.add_argument("--verbose", action="store_true", help="Print detector/tracker debug logs.")
    parser.add_argument("--reset-events", action="store_true", help="Clear the JSONL event file before running.")
    args = parser.parse_args()

    config = load_config(args.config, base_dir=base_dir)
    if args.model:
        config["model_path"] = args.model
    if args.video:
        config["source"] = args.video
    if args.source:
        config["source"] = args.source
    if args.camera:
        config["source"] = 0
    if args.output:
        config["output_video"] = args.output
        config["save_video"] = True
    if args.max_frames:
        config["max_frames"] = args.max_frames
    if args.no_display:
        config["display"] = False

    model_path = config["model_path"]
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行项目1导出 yolov8n.onnx")
        return
    
    source = config["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, str) and "://" not in source and not os.path.exists(source):
        print(f"❌ 视频源不存在: {source}")
        return

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {source}")
        return

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
    anomaly = build_anomaly_engine(config, fps)
    roi_filter = RoiFilter(**config["roi"])
    if args.reset_events and os.path.exists(config["event_jsonl"]):
        os.remove(config["event_jsonl"])
    event_writer = EventWriter(
        event_jsonl=config["event_jsonl"],
        snapshot_dir=config["snapshot_dir"],
        camera_id=config["camera_id"],
    )

    print("🚀 多目标跟踪与异常行为识别系统 - 交付版")
    print(f"   camera_id={config['camera_id']}")
    print(f"   source={source}")
    print(f"   model={model_path}")
    print(f"   events={config['event_jsonl']}")
    
    frame_id = 0
    writer = None
    start_time = time.time()
    dropped_reads = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str) and "://" in source and dropped_reads < config["reconnect_attempts"]:
                dropped_reads += 1
                print(f"⚠️ 读取失败，尝试重连 {dropped_reads}/{config['reconnect_attempts']}")
                cap.release()
                time.sleep(config["reconnect_delay_sec"])
                cap = open_capture(source)
                continue
            print("✅ 视频源结束或不可读取")
            break

        frame_id += 1
        
        tracked = mot.process_frame(frame)
        tracked = roi_filter.filter_objects(tracked)
        
        alerts = anomaly.update(tracked, frame_id) if config["anomaly"]["enabled"] else []
        
        frame = draw_tracks(frame, tracked)
        frame = draw_alerts(frame, alerts)
        frame = roi_filter.draw(frame)
        event_writer.write_alerts(alerts, frame, frame_id, frame_id / fps)
        
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if config["save_video"]:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                os.makedirs(os.path.dirname(config["output_video"]), exist_ok=True)
                writer = cv2.VideoWriter(config["output_video"], fourcc, fps, (w, h))
            writer.write(frame)

        if config["display"]:
            cv2.imshow("MOT + Anomaly Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if config["log_interval"] and frame_id % config["log_interval"] == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[RUN] frame={frame_id} tracks={len(tracked)} alerts={len(alerts)} "
                f"speed={frame_id / elapsed:.2f} fps"
            )

        if config["max_frames"] and frame_id >= config["max_frames"]:
            print(f"✅ 已处理 {frame_id} 帧")
            break
    
    cap.release()
    if writer is not None:
        writer.release()
    if config["display"]:
        cv2.destroyAllWindows()
    elapsed = max(time.time() - start_time, 1e-6)
    print(f"✅ 程序退出，共处理 {frame_id} 帧，平均 {frame_id / elapsed:.2f} FPS")

if __name__ == "__main__":
    main()
