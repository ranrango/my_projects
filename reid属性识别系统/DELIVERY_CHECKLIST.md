# 交付检查清单

## 必备文件

```text
README.md
DEPLOYMENT.md
DELIVERY_CHECKLIST.md
Dockerfile
docker-compose.yml
.env.example
reid_baseline/
checkpoints/reid_baseline.pth
deploy/gallery_index.npz
```

## 交付前检查

1. `python -m py_compile reid_baseline/deploy/*.py reid_baseline/*.py`
2. `python -m reid_baseline.deploy.build_gallery ...`
3. `python -m reid_baseline.deploy.search ...`
4. `uvicorn reid_baseline.deploy.api:app --host 127.0.0.1 --port 8000`
5. `curl http://127.0.0.1:8000/health`
6. `curl -X POST "http://127.0.0.1:8000/search?topk=5" -F "file=@query.jpg"`

## 性能建议

- GPU 部署：`REID_DEVICE=cuda`，图库构建和在线推理都明显更快。
- CPU 部署：适合小并发和小图库，建议把 `REID_MAX_TOPK` 控制在 50 以内。
- 中小图库：当前 Numpy 矩阵检索足够，已使用 `argpartition` 做 top-k。
- 大图库：超过几十万张建议切换 FAISS/HNSW 索引。
- 索引更新：新图库构建到临时文件，验证后替换 `REID_INDEX`，调用 `/reload`。
- 安全：生产环境设置 `REID_API_KEY`，默认不暴露服务器绝对路径。

## 业务边界

本服务只做行人裁剪图的 ReID 检索，不做：

- 视频流拉取
- 行人检测
- 多目标跟踪
- 用户/权限系统
- 图片对象存储
- 结果人工复核工作流

这些应由上游检测/跟踪服务和业务后端组合完成。

