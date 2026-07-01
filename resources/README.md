# 外部资源说明

本目录用于说明运行演示所需的外部资源。不要把大文件或隐私数据提交到 Git。

## 模型权重

- 检测项目：将 `yolov8n.onnx` 放到对应项目根目录，或按 README 使用 `onnx_out.py` 导出。
- MOT 项目：将 `yolov8n.onnx` 放到 `多目标跟踪与异常行为识别系统（进阶）/models/`。
- ReID 项目：将 `reid_baseline.pth` 放到 `reid属性识别系统/checkpoints/`。
- 门禁项目：当前基线不需要深度学习模型；生产升级可替换为 ArcFace 或 MobileFaceNet ONNX。

## 样例视频和数据

- 检测视频放到对应项目根目录，命名为 `test.mp4` 或运行时通过 `--video` 指定。
- MOT 视频放到对应项目根目录，命名为 `test.mp4` 或在配置文件中指定。
- ReID 数据推荐使用 Market1501；无真实数据时可运行 toy 数据生成脚本。
- 门禁人脸图片只放本地 `enrolled_faces/`，不要提交到仓库。

## 隐私规则

- 不提交真实人脸、监控原视频、访问日志、训练数据、API Key、RTSP URL 或本机绝对路径。
- 可公开展示的截图和 SVG 放入 `assets/`。
