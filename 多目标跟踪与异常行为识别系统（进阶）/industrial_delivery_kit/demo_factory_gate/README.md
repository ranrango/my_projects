# 演示场景：厂区出入口车辆通行监控

## 场景目标

厂区门口摄像头需要识别并跟踪进入画面的车辆，限定重点监控区域，输出带框视频和结构化事件文件。该场景适合园区门岗、物流通道、仓库出入口、停车场入口的 PoC 演示。

## 当前演示能力

- 检测 `person/car/bus/truck`。
- 对车辆进行跨帧 ID 跟踪。
- 绘制黄色 ROI 重点监控区域。
- 输出标注视频。
- 输出 JSONL 事件文件。当前测试视频没有触发行人异常，所以事件文件为空，这是正常结果。

## 运行命令

```bash
cd /path/to/多目标跟踪与异常行为识别系统（进阶）
python3 industrial_delivery_kit/src/edge_runtime.py \
  --config industrial_delivery_kit/demo_factory_gate/factory_gate_config.json \
  --max-frames 8
```

## 输出

```text
industrial_delivery_kit/demo_factory_gate/runs/industrial_output.mp4
industrial_delivery_kit/demo_factory_gate/runs/industrial_events.jsonl
industrial_delivery_kit/demo_factory_gate/runs/snapshots/
```

## 生产落地补充

如果要做真正的厂区车辆管理，需要补充现场数据并训练专用类别，例如 `forklift`、`truck`、`worker`、`helmet`、`vest`，再接入车牌识别、道闸系统或告警平台。
