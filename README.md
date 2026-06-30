# CV Projects — 计算机视觉工程项目集

> 四个可落地的边缘视觉系统，覆盖**目标检测 → 多目标跟踪 → 行人重识别 → 端侧门禁**完整技术链路。
> 所有项目均可在纯 CPU 环境运行，无需 GPU。

---

## 项目目录

| # | 项目 | 核心技术 | 依赖 |
|---|------|----------|------|
| 1 | [基于YOLOv8的轻量化实时行人与车辆检测系统](#1-基于yolov8的轻量化实时行人与车辆检测系统) | YOLOv8n · ONNX · OpenCV DNN | opencv, numpy |
| 2 | [多目标跟踪与异常行为识别系统（进阶）](#2-多目标跟踪与异常行为识别系统进阶) | ByteTrack · Kalman · 异常规则引擎 | opencv, numpy, scipy |
| 3 | [reid属性识别系统](#3-reid属性识别系统) | ResNet50 · Triplet Loss · FastAPI | torch, torchvision, fastapi |
| 4 | [端侧人脸识别门禁系统（工程+安全）](#4-端侧人脸识别门禁系统工程安全) | Haar · LBP · 活体检测 · 审计日志 | opencv, numpy |

---

## 1. 基于YOLOv8的轻量化实时行人与车辆检测系统

**目标**：在 CPU 上实时检测行人、小汽车、公交车、卡车，适合边缘摄像头部署。

**技术要点**
- YOLOv8n 导出为 ONNX，通过 OpenCV DNN 推理，**零 PyTorch 依赖**
- Letter-box 预处理保持宽高比，NMS 后处理
- 支持摄像头实时流、本地视频文件、无界面批处理三种运行模式

**性能参考**：Intel i5 CPU，640×480 输入，约 20–25 FPS

```bash
cd 基于YOLOv8的轻量化实时行人与车辆检测系统
pip install -r requirements.txt
python run_video.py --video test.mp4 --output output.mp4
```

---

## 2. 多目标跟踪与异常行为识别系统（进阶）

**目标**：在检测基础上实现跨帧 ID 跟踪，并输出人群聚集、跌倒、徘徊等异常事件。

**技术要点**
- 手写 ByteTrack：Kalman 滤波预测 + 匈牙利匹配 + 短时丢失缓冲
- 异常规则引擎：可配置持续时间、冷却时间、ROI 多边形过滤
- 工程化输出：JSONL 结构化事件 + 报警截图留证 + 标注视频
- 支持 RTSP 断线重连、无界面生产模式

附带 `industrial_delivery_kit/`：厂区门口场景 PoC 完整交付套件（配置、演示、对比报告）。

```bash
cd 多目标跟踪与异常行为识别系统（进阶）
pip install -r requirements.txt
python validate_system.py                        # 全链路验收
python run_video.py --config config/default_config.json --no-display
```

---

## 3. reid属性识别系统

**目标**：基于 Market-1501 数据集训练行人重识别模型，支持图库检索和 API 服务部署。

**技术要点**
- ResNet50 backbone，输出 512 维 ReID embedding
- 联合交叉熵 + Batch Triplet Loss 训练
- 评估指标：mAP、Rank-1/5/10
- 完整部署链路：TorchScript 导出 → 图库建索引 → FastAPI 检索服务
- Docker / Docker Compose 一键启动

```bash
cd reid属性识别系统
pip install -r reid_baseline/requirements.txt

# 用 toy 数据快速验证链路（不需要真实数据集）
python3 -m reid_baseline.scripts.create_toy_market1501 --output data/toy_market1501
python3 -m reid_baseline.train --data-root data/toy_market1501 --epochs 1 --device cpu --no-pretrained
python3 -m reid_baseline.inference.evaluate_reid --data-root data/toy_market1501 --checkpoint checkpoints/reid_baseline.pth
```

---

## 4. 端侧人脸识别门禁系统（工程+安全）

**目标**：纯 CPU 门禁系统，从人脸注册到访问授权到安全审计，完整工程闭环。

**技术要点**

| 模块 | 实现 |
|------|------|
| 人脸检测 | OpenCV Haar Cascade，零外部模型 |
| 特征提取 | 4×4 网格 LBP 直方图，944 维，L2 归一化 |
| 活体检测 | Laplacian 方差，拒绝印刷图片翻拍 |
| 安全防护 | 连续失败锁定 + 授权冷却 + 每次事件 JSONL 留证 |
| 注册工具 | 目录批量注册 / 摄像头实时采集，两种模式 |

**全链路自检 11/11 通过**（不需要摄像头）：

```bash
cd 端侧人脸识别门禁系统（工程+安全）
pip install -r requirements.txt
python3 validate_system.py

# 注册人脸（放入 enrolled_faces/<姓名>/*.jpg 后执行）
python3 enroll.py build

# 启动门禁
python3 run_gate.py --source test.mp4
```

> 升级路径：将 `FaceEmbedder.embed()` 替换为 ArcFace / MobileFaceNet ONNX 模型，接口不变，识别率可达 99%+。

---

## 技术栈总览

```
目标检测        YOLOv8n (ONNX) + OpenCV DNN
多目标跟踪      ByteTrack (手写) + Kalman Filter
行人重识别      ResNet50 + Triplet Loss + FastAPI
人脸识别        Haar Cascade + LBP + Laplacian Liveness
推理框架        ONNX Runtime / OpenCV DNN（无 GPU 要求）
部署            Docker · TorchScript · REST API
```

## 安装

各项目依赖相互独立，按需进入对应目录安装：

```bash
pip install -r <项目目录>/requirements.txt
```

最低公共依赖：`opencv-python >= 4.8`，`numpy >= 1.21`

## 说明

- 模型权重（`*.pt`、`*.onnx`、`*.pth`）、视频文件、数据集均不含在仓库中，需自行下载或导出
- 人脸图片（`enrolled_faces/`）和访问日志（`audit_logs/`）属于隐私数据，已在 `.gitignore` 中排除
