# 性能与落地建议

## 当前实现适用范围

当前部署版使用：

- PyTorch/torchvision 做特征提取
- L2-normalized embedding
- Numpy 矩阵乘法计算余弦相似度
- `argpartition` 做 top-k，避免全量排序
- `.npz` 保存图库特征、路径和业务 ID

适合：

- 几千到几十万级图库
- 单机 GPU/CPU 部署
- 低到中等并发的图片检索服务

## 性能瓶颈

1. 特征提取：主要受模型和 GPU/CPU 影响。
2. 图库检索：主要受图库数量和 embedding 维度影响。
3. 图片 IO：图库构建时会读取大量小图，磁盘 IO 会影响速度。
4. API 上传：图片过大或并发过高会影响延迟。

## 线上参数建议

```bash
REID_DEVICE=cuda
REID_DEFAULT_TOPK=5
REID_MAX_TOPK=50
REID_MAX_UPLOAD_MB=10
REID_EXPOSE_PATHS=false
REID_API_KEY=your-production-token
```

阈值建议从业务数据上调，不要直接照搬 Market1501：

```bash
REID_SCORE_THRESHOLD=0.65
```

粗略经验：

- `score >= 0.80`：通常很相似
- `0.65 <= score < 0.80`：建议人工复核或结合轨迹
- `score < 0.65`：通常不够可靠

实际阈值必须用你的摄像头、角度、清晰度、服装变化数据验证。

## 大图库升级路线

当图库达到几十万到百万级时，建议替换 `GalleryIndex.search` 为 FAISS：

```text
features -> faiss.IndexFlatIP / IndexIVFFlat / HNSW
query feature -> faiss.search(topk)
```

升级优先级：

1. 10 万以内：当前 Numpy 方案通常够用。
2. 10 万到 100 万：FAISS `IndexFlatIP` 或 HNSW。
3. 百万以上：IVF/PQ 或专门向量数据库。

## 工程落地架构

```text
摄像头/视频
  -> 检测模型裁剪行人图
  -> 跟踪服务生成 track_id
  -> ReID API 提取特征/检索图库
  -> 业务后端保存结果
  -> 人工复核/告警/看板
```

这个项目承担 ReID API 这一层。检测、跟踪、存储、鉴权和前端看板建议由独立服务负责。

## 更新图库

建议使用“双索引”更新策略：

1. 新图库构建到 `deploy/gallery_index.next.npz`
2. 用几张 query 做命令行检索验证
3. 替换为正式索引路径
4. 调用 `/reload`

这样避免线上服务在构建索引时不可用。

## 可靠性建议

- API 前面加 Nginx 或 API Gateway。
- 配置 `REID_API_KEY`，不要裸露服务。
- 生产环境不要打开 `REID_EXPOSE_PATHS`。
- 用 supervisor/systemd/docker compose 保活。
- 记录 query 图片 ID、top-k 结果、score、latency，方便回溯。
- 保留 checkpoint、配置文件和 gallery 索引版本号。

