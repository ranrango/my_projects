# 工业部署方案

## 1. 单机边缘盒子

适用：

- 1 到 8 路摄像头。
- 工厂、园区、仓库、本地安防。
- 低延迟、本地留证、弱网环境。

推荐：

- NVIDIA Jetson / 工控机 + NVIDIA GPU：DeepStream + TensorRT。
- Intel CPU/iGPU/NPU：OpenVINO Runtime 或 OpenVINO Model Server。
- 普通 CPU 试点：ONNX + OpenCV DNN 或 ONNX Runtime。

架构：

```text
RTSP/NVR -> 解码 -> 检测 -> 跟踪 -> 规则引擎 -> 事件 JSONL/MQTT/HTTP -> 平台
                                  -> 截图/视频留证 -> 本地磁盘/NAS
```

## 2. GPU 服务器集中推理

适用：

- 多路摄像头集中接入。
- 多模型级联。
- 要求统一模型版本和水平扩容。

推荐：

- NVIDIA Triton Inference Server。
- Ensemble pipeline：preprocess -> detector -> postprocess -> optional pose/reid。
- Kafka/MQTT/HTTP 接入业务系统。

架构：

```text
视频接入服务 -> 抽帧/批处理 -> Triton ensemble -> 事件规则服务 -> 数据库/消息队列/告警平台
```

## 3. DeepStream 路线

适用：

- NVIDIA GPU/Jetson。
- 多路视频吞吐优先。
- 需要硬件解码、零拷贝、端到端视频 analytics。

推荐组件：

- nvurisrcbin/nvstreammux：多路输入。
- nvinfer：TensorRT 模型推理。
- nvtracker：内置跟踪。
- nvdsanalytics：ROI、线穿越、区域统计。
- nvmsgbroker：事件上云。
- smart record：报警前后视频片段留证。

## 4. OpenVINO 路线

适用：

- Intel CPU/iGPU/NPU 工控机。
- 不希望依赖 NVIDIA GPU。
- 对部署成本敏感。

推荐：

- ONNX/IR 模型转换。
- FP16/INT8 优化。
- OpenVINO Runtime 单机部署，或 OpenVINO Model Server 服务化。

## 5. 当前项目落地方式

当前代码适合以下交付形态：

- PoC：直接运行 `run_video.py` 或 `industrial_delivery_kit/src/edge_runtime.py`。
- 小规模试点：每路摄像头一个进程，事件写 JSONL 或转 HTTP/MQTT。
- 下一阶段生产：将检测器替换为 TensorRT/OpenVINO backend，将事件写入消息队列。

## 6. 部署检查表

- 模型文件固定版本，记录 md5/sha256。
- 配置文件纳入版本管理。
- 每路摄像头独立 camera_id。
- RTSP 断线重连。
- 本地磁盘容量保护。
- 日志轮转。
- 报警截图和视频片段留存周期。
- 误报/漏报反馈入口。
- 上线前压测 FPS、CPU、GPU、内存、温度和端到端延迟。
