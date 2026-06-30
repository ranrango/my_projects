# 工业界交付方案说明

这个文件夹用于把当前“多目标跟踪与异常行为识别系统”升级成工业现场可评审、可部署、可验收的方案包。

## 面向需求

- 多路摄像头或 RTSP 视频接入。
- 行人、车辆、叉车、PPE 等目标检测。
- 多目标跟踪与跨帧 ID 保持。
- 人群聚集、跌倒、徘徊、禁区闯入、车辆违规停留等事件。
- 边缘端实时推理、事件结构化输出、截图/视频留证。
- 后续支持模型微调、版本管理、数据闭环。

## 文件说明

- `docs/algorithm_selection.md`：工业界最新算法选型建议。
- `docs/deployment_playbook.md`：边缘端、GPU 服务器、Intel CPU/NPU 的部署路线。
- `configs/industrial_config.json`：工业现场配置模板。
- `src/edge_runtime.py`：可运行参考入口，复用本项目已有检测/跟踪/异常模块。
- `src/solution_matrix.py`：按需求输出算法与部署建议。
- `validate_delivery.py`：一键验收脚本。

## 快速验收

在上一级项目目录执行：

```bash
python3 industrial_delivery_kit/validate_delivery.py
python3 industrial_delivery_kit/src/edge_runtime.py --config industrial_delivery_kit/configs/industrial_config.json --max-frames 3
python3 industrial_delivery_kit/src/solution_matrix.py
```

产物默认输出到：

```text
industrial_delivery_kit/runs/
```

## 交付定位

这里提供的是“工程可运行 + 工业方案可落地”的交付骨架。生产上线前仍建议补现场数据标注、模型微调、报警复核闭环和硬件压测。
