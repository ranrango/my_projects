# 作品集交付说明

这个仓库包含四个计算机视觉工程项目，覆盖目标检测、多目标跟踪、行人重识别和端侧门禁。它面向作品集展示、面试讲解和 PoC 交流，不直接携带私有数据、模型权重或真实人脸。

## 5 分钟快速浏览

1. 看根目录 `README.md` 的项目矩阵和成果图。
2. 看 `assets/` 下的四张展示图。
3. 重点关注 ReID 的指标表和 MOT 的工业交付套件说明。

## 15 分钟技术深读

1. 运行 `python3 scripts/validate_all.py` 查看基础可运行性。
2. 阅读每个项目 README 的「技术亮点」「结果指标」「边界说明」。
3. 选择一个项目进入源码：检测看 `detector.py`，MOT 看 `core/bytetracker.py`，ReID 看 `reid_baseline/train.py`，门禁看 `face_gate/security/access_controller.py`。

## 30 分钟演示路径

1. 按 `resources/README.md` 放置模型、视频或数据。
2. 检测项目运行 `python run_video.py --video test.mp4 --no-display`。
3. MOT 项目运行 `python validate_system.py` 和 `python run_video.py --config config/default_config.json --no-display --max-frames 30`。
4. ReID 项目运行 toy 数据链路，验证训练、评估和检索服务入口。
5. 门禁项目运行 `python validate_system.py`，检查注册、检索、安全和审计闭环。

## 交付边界

- 不提交模型权重、数据集、真实门禁照片、RTSP 账号、访问日志和云服务密钥。
- 当前检测和门禁项目偏 CPU 工程基线，适合展示系统设计和端侧部署思路。
- 工业上线前需要现场数据标注、真实指标评估、推理加速、日志轮转和人工复核流程。
