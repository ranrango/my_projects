import os
import tempfile
import time
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from reid_baseline.deploy.inferencer import GalleryIndex, ReIDInferencer
from reid_baseline.deploy.settings import ApiSettings


settings = ApiSettings()

app = FastAPI(title="ReID Retrieval API")
inferencer = None
gallery_index = None


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_runtime():
    global inferencer, gallery_index
    inferencer = ReIDInferencer(settings.checkpoint_path, device=settings.device)
    gallery_index = GalleryIndex.load(settings.index_path)


@app.on_event("startup")
def startup():
    load_runtime()


@app.get("/health")
def health():
    checkpoint = settings.checkpoint_path if settings.expose_paths else os.path.basename(settings.checkpoint_path)
    index = settings.index_path if settings.expose_paths else os.path.basename(settings.index_path)
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "index": index,
        "device": str(inferencer.device) if inferencer else settings.device,
        "gallery_size": int(len(gallery_index.paths)) if gallery_index else 0,
        "expose_paths": settings.expose_paths,
        "max_topk": settings.max_topk,
    }


@app.post("/reload", dependencies=[Depends(require_api_key)])
def reload_index():
    load_runtime()
    return {"status": "ok", "gallery_size": int(len(gallery_index.paths))}


@app.post("/search", dependencies=[Depends(require_api_key)])
async def search(file: UploadFile = File(...), topk: Optional[int] = None, threshold: Optional[float] = None):
    started = time.perf_counter()
    topk = settings.default_topk if topk is None else topk
    if topk < 1 or topk > settings.max_topk:
        raise HTTPException(status_code=400, detail=f"topk must be between 1 and {settings.max_topk}")

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large, max {settings.max_upload_mb} MB")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            feature = inferencer.extract_image(tmp.name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid image file") from exc
    final_threshold = settings.score_threshold if threshold is None else threshold
    results = gallery_index.search(
        feature,
        topk=topk,
        threshold=final_threshold,
        include_path=settings.expose_paths,
    )
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "threshold": final_threshold,
    }
