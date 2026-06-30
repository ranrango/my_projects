# 多目标跟踪与异常行为识别系统

这是一个可交付运行的轻量级边缘视觉应用，基于 YOLOv8 ONNX + OpenCV DNN + 简化 ByteTrack，支持视频/摄像头/RTSP 输入、目标跟踪、ROI 过滤、异常事件输出、报警截图留证和结果视频保存。

## 已具备能力

- 行人/车辆检测：支持 COCO 类别 person、car、bus、truck。
- 多目标跟踪：跨帧保持 track_id，支持短时丢失缓冲。
- 异常识别：人群聚集、跌倒、徘徊，支持规则开关、持续时间和报警冷却。
- 工程化输出：JSONL 事件、报警截图、可选标注视频。
- 生产入口：配置文件驱动，支持无界面运行、RTSP 重连、运行状态日志。
- 交付验收：提供 `validate_system.py` 一键验证核心链路。

## 目录结构

```text
.
├── anomaly/                 # 异常规则引擎
├── config/default_config.json
├── core/                    # 检测、跟踪、MOT 封装
├── models/yolov8n.onnx
├── runs/                    # 运行产物，自动生成
├── utils/                   # 配置、ROI、事件、可视化
├── run_video.py             # 生产运行入口
├── validate_system.py       # 交付验收脚本
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 本地验收

```bash
python3 validate_system.py
python3 run_video.py --config config/default_config.json --no-display --max-frames 30 --reset-events
```

运行后产物默认在：

- `runs/events.jsonl`：报警事件，每行一个 JSON。
- `runs/snapshots/`：报警截图。
- `runs/demo_output.mp4`：带框和报警标注的视频。

## RTSP/摄像头运行

使用 RTSP：

```bash
python3 run_video.py --source "rtsp://user:password@ip:554/stream1" --no-display
```

使用本机摄像头：

```bash
python3 run_video.py --camera
```

指定输出视频：

```bash
python3 run_video.py --source test.mp4 --output runs/site_output.mp4 --no-display
```

## 配置说明

主要配置在 `config/default_config.json`：

- `camera_id`：摄像头/点位编号，会写入事件。
- `model_path`：ONNX 模型路径。
- `source`：输入源，可以是视频、摄像头编号或 RTSP URL。
- `event_jsonl`：事件输出文件。
- `snapshot_dir`：截图目录。
- `detector`：输入尺寸、置信度、NMS 阈值、检测类别。
- `tracker`：跟踪阈值、缓存帧数、匹配阈值。
- `roi`：是否启用多边形区域过滤。
- `anomaly`：异常规则开关、阈值、持续时间和冷却时间。

ROI 示例：

```json
"roi": {
  "enabled": true,
  "polygon": [[100, 80], [580, 80], [620, 340], [80, 340]]
}
```

## 事件格式

`runs/events.jsonl` 示例：

```json
{"camera_id":"demo_camera_001","event_type":"crowd","frame_id":120,"timestamp_sec":4.0,"wall_time":"2026-05-28 10:00:00","location":[320,180],"track_id":null,"count":3,"snapshot_path":"runs/snapshots/000120_crowd.jpg"}
```

## 工业落地建议

当前版本已经适合作为现场 PoC 和小规模试点。真正生产上线前，建议补充现场数据微调、姿态模型跌倒识别、ReID 外观特征、多路摄像头调度、Web 管理后台和人工复核闭环。
