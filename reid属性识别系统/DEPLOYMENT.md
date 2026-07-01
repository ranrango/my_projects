# ReID 部署说明

这个项目部署的是“行人重识别检索服务”：输入一张已经裁剪好的行人图，输出图库中最相似的行人图片。实际业务里通常需要上游先用 YOLO/RT-DETR/检测跟踪模型裁出行人框，再把裁剪图传给本服务。

## 1. 部署产物

训练完成后至少需要这两个文件：

```text
checkpoints/reid_baseline.pth   # 训练好的 ReID checkpoint
deploy/gallery_index.npz        # 目标图库特征索引
```

可选导出：

```text
deploy/reid_feature_extractor.pt # TorchScript 特征提取模型
```

## 2. 安装依赖

```bash
cd /root/autodl-tmp/reid_project
python -m pip install -r reid_baseline/requirements.txt
```

本机调试时把路径换成项目目录即可。

## 3. 训练模型

云服务器上已经跑通过的 10 轮训练命令：

```bash
python -m reid_baseline.train \
  --data-root /root/autodl-tmp/datasets/Market-1501-v15.09.15 \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda
```

训练完成后会生成：

```text
checkpoints/reid_baseline.pth
```

本次 10 轮全量评估结果：

```text
mAP:    0.5588
Rank-1: 0.7957
Rank-5: 0.9121
Rank-10: 0.9362
```

## 4. 构建图库索引

图库目录应放行人裁剪图，可以是扁平目录，也可以有子目录：

```text
gallery/
  person_a_001.jpg
  camera_1/person_b_003.jpg
```

构建索引：

```bash
python -m reid_baseline.deploy.build_gallery \
  --checkpoint checkpoints/reid_baseline.pth \
  --gallery-dir /path/to/person/gallery \
  --output deploy/gallery_index.npz \
  --batch-size 128 \
  --device cuda
```

如果只是用 Market1501 gallery 做测试：

```bash
python -m reid_baseline.deploy.build_gallery \
  --checkpoint checkpoints/reid_baseline.pth \
  --gallery-dir /root/autodl-tmp/datasets/Market-1501-v15.09.15/bounding_box_test \
  --output deploy/gallery_index.npz \
  --batch-size 128 \
  --device cuda \
  --skip-invalid-market1501
```

## 5. 命令行检索

```bash
python -m reid_baseline.deploy.search \
  --checkpoint checkpoints/reid_baseline.pth \
  --index deploy/gallery_index.npz \
  --query /path/to/query/person.jpg \
  --topk 5 \
  --device cuda
```

输出示例：

```text
Rank-1: score=0.8732 path=/path/to/gallery/xxx.jpg
Rank-2: score=0.8215 path=/path/to/gallery/yyy.jpg
```

score 是 L2 归一化 embedding 的余弦相似度，越大越相似。

## 6. 启动 API 服务

```bash
export REID_CHECKPOINT=checkpoints/reid_baseline.pth
export REID_INDEX=deploy/gallery_index.npz
export REID_DEVICE=cuda
export REID_EXPOSE_PATHS=false
export REID_API_KEY="<set-a-strong-random-token>"

uvicorn reid_baseline.deploy.api:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

检索接口：

```bash
curl -X POST "http://127.0.0.1:8000/search?topk=5" \
  -H "X-API-Key: <set-a-strong-random-token>" \
  -F "file=@/path/to/query/person.jpg"
```

返回格式：

```json
{
  "results": [
    {"rank": 1, "id": "camera_1/a.jpg", "score": 0.8732},
    {"rank": 2, "id": "camera_2/b.jpg", "score": 0.8215}
  ],
  "latency_ms": 38.42,
  "threshold": null
}
```

默认不会返回服务器绝对路径。如果内网调试需要返回路径，可设置：

```bash
export REID_EXPOSE_PATHS=true
```

更新图库索引后可热加载：

```bash
curl -X POST http://127.0.0.1:8000/reload -H "X-API-Key: change-me"
```

## 6.1 Docker 部署

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

将模型和索引放到：

```text
checkpoints/reid_baseline.pth
deploy/gallery_index.npz
```

## 7. 导出 TorchScript

如果要给非 Python 服务加载特征提取模型，可以导出 TorchScript：

```bash
python -m reid_baseline.deploy.export_torchscript \
  --checkpoint checkpoints/reid_baseline.pth \
  --output deploy/reid_feature_extractor.pt \
  --device cpu
```

导出的模型输入为：

```text
shape: [N, 3, 256, 128]
format: RGB, float tensor, ImageNet mean/std normalize
output: [N, 512] L2-normalized embedding
```

## 8. 线上接入建议

1. 上游检测模型先裁剪行人图，尽量保留完整人体，少包含背景。
2. 新增或更新图库图片后，重新运行 `build_gallery` 生成索引。
3. 检索阈值需要用业务数据调，建议先观察正负样本 score 分布，再定阈值。
4. API 服务只做 ReID 检索，不负责身份注册、权限、数据库和图片存储；这些应由业务服务管理。
5. 当前索引是内存内 Numpy 矩阵，适合中小图库。图库到几十万级后建议换 FAISS。
