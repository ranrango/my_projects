# 端侧人脸识别门禁系统（工程+安全）

> 纯 CPU 门禁系统：OpenCV Haar 检测 + LBP 特征识别 + 纹理活体检测 + 安全审计日志。零深度学习依赖，开箱即用。

## 系统架构

```
run_gate.py          # 生产运行入口（摄像头/视频/RTSP）
enroll.py            # 人脸注册 CLI（目录批量 / 摄像头采集）
validate_system.py   # 交付验收脚本（不需要摄像头）
face_gate/
  detection/
    detector.py      # Haar 人脸检测器
    liveness.py      # Laplacian 方差活体检测
  recognition/
    embedder.py      # LBP 直方图特征提取（L2归一化）
    gallery.py       # 人脸库管理（npz 索引）
  security/
    guard.py         # 失败计数 + 锁定逻辑
    access_controller.py  # 授权决策 + 门状态管理
  utils/
    audit_logger.py  # JSONL 审计日志 + 截图留证
    config.py        # 配置加载
    visualizer.py    # 画框 / 状态渲染
config/
  default_config.json
enrolled_faces/      # 注册人脸图片（git 忽略，勿提交）
audit_logs/          # 访问日志 + 截图（git 忽略，勿提交）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 注册人脸

**方式 A — 目录批量注册**（每人建一个子目录，放入至少 3 张照片）:

```
enrolled_faces/
  Alice/
    001.jpg
    002.jpg
    003.jpg
  Bob/
    001.jpg
    ...
```

```bash
python enroll.py build
```

**方式 B — 摄像头实时采集**:

```bash
python enroll.py capture --identity Alice --count 10
# 按 SPACE 采集，Q 退出，自动重建图库
```

**查看已注册人员**:

```bash
python enroll.py list
```

### 3. 启动门禁

```bash
# 默认摄像头
python run_gate.py

# 指定视频文件
python run_gate.py --source test.mp4

# 无头运行（服务器）
python run_gate.py --source test.mp4 --no-display
```

### 4. 交付验收

```bash
python validate_system.py
```

不需要摄像头，全链路自检（检测 → 活体 → 嵌入 → 检索 → 安全 → 审计）。

## 安全机制

| 机制 | 说明 |
|------|------|
| 活体检测 | Laplacian 方差检测，拒绝印刷图片 |
| 失败锁定 | 连续 5 次失败后锁定 60 秒 |
| 冷却限流 | 同一人连续授权需间隔 5 秒 |
| 审计日志 | 每次 GRANT / DENY / 活体失败均写入 JSONL + 截图 |
| 隐私保护 | `enrolled_faces/` 和 `audit_logs/` 均在 `.gitignore` 中，绝不入库 |

## 审计日志格式

`audit_logs/access_log.jsonl` 每行一条记录：

```json
{"timestamp":"2026-06-30T10:00:00.123","event_type":"grant","identity":"Alice","frame_id":42,"score":0.812}
{"timestamp":"2026-06-30T10:00:05.456","event_type":"deny","identity":"UNKNOWN","frame_id":155,"score":0.21,"lockout_triggered":false}
```

## 配置说明

主要配置在 `config/default_config.json`：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `recognition.match_threshold` | 识别相似度阈值，越高越严格 | `0.55` |
| `liveness.texture_var_threshold` | 活体纹理方差阈值 | `200.0` |
| `security.max_failed_attempts` | 触发锁定的失败次数 | `5` |
| `security.lockout_duration_sec` | 锁定时长（秒） | `60` |
| `access.gate_open_duration_sec` | 门开启持续时间（秒） | `3` |
| `access.cooldown_sec` | 同一人再次授权的冷却时间 | `5` |

## 算法说明

- **检测**：OpenCV Haar Cascade，CPU 实时，无外部模型文件
- **特征提取**：4×4 网格 LBP 直方图拼接，944 维，L2 归一化
- **比对**：余弦相似度（= 单位向量点积）
- **活体**：Laplacian 方差，印刷/屏幕翻拍方差低于真实人脸

> 生产升级：将 `FaceEmbedder.embed()` 替换为 ArcFace / MobileFaceNet ONNX 模型，接口保持不变，准确率可提升至 99%+。

## 依赖

```
opencv-python>=4.8.0
numpy>=1.21.0
```

仅依赖 OpenCV 和 NumPy，无 PyTorch / TensorFlow 等重型依赖。

## 许可

MIT
