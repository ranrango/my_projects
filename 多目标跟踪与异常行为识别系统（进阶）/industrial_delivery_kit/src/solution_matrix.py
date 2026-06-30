import json


SOLUTION_MATRIX = [
    {
        "need": "行人/车辆/叉车/PPE 检测",
        "recommended_algorithm": "YOLO11/YOLOv8 for edge, RT-DETR/DETR family for higher accuracy",
        "current_project_mapping": "YOLOv8 ONNX detector; replace weights with site-trained model",
        "deployment": "CPU PoC: OpenCV DNN; Intel: OpenVINO; NVIDIA: TensorRT/DeepStream; Server: Triton",
    },
    {
        "need": "多目标跟踪与 ID 保持",
        "recommended_algorithm": "ByteTrack for real-time scenes; BoT-SORT/OC-SORT + ReID for occlusion",
        "current_project_mapping": "Simplified ByteTrack with Kalman prediction and IoU matching",
        "deployment": "Run after detector in same edge process; use ReID model as optional secondary inference",
    },
    {
        "need": "禁区闯入",
        "recommended_algorithm": "Detection + polygon ROI + duration/cooldown rules",
        "current_project_mapping": "RoiFilter + event writer",
        "deployment": "Configure per camera; emit JSONL/MQTT/HTTP events",
    },
    {
        "need": "人群聚集",
        "recommended_algorithm": "Person tracks + distance clustering + temporal confirmation",
        "current_project_mapping": "AnomalyEngine crowd rule",
        "deployment": "Edge rule engine; tune threshold by camera angle and scene scale",
    },
    {
        "need": "跌倒识别",
        "recommended_algorithm": "YOLO-pose/RTMPose + temporal rules; bbox heuristic only as first-stage warning",
        "current_project_mapping": "BBox ratio and vertical speed heuristic",
        "deployment": "Production should add pose model as second-stage inference for person tracks",
    },
    {
        "need": "徘徊/违规停留",
        "recommended_algorithm": "Track trajectory area + dwell time + ROI semantics",
        "current_project_mapping": "AnomalyEngine loitering rule",
        "deployment": "Edge rule engine with per-zone dwell-time config",
    },
    {
        "need": "事件留证与平台对接",
        "recommended_algorithm": "Structured event schema + snapshot/video evidence + message queue",
        "current_project_mapping": "EventWriter JSONL + snapshots + annotated video",
        "deployment": "PoC JSONL; production MQTT/Kafka/HTTP webhook",
    },
]


def main():
    print(json.dumps(SOLUTION_MATRIX, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
