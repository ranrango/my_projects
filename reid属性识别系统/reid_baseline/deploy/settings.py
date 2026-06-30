import os
from dataclasses import dataclass
from typing import Optional


def _get_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ApiSettings:
    checkpoint_path: str = os.environ.get("REID_CHECKPOINT", "checkpoints/reid_baseline.pth")
    index_path: str = os.environ.get("REID_INDEX", "deploy/gallery_index.npz")
    device: str = os.environ.get("REID_DEVICE", "auto")
    api_key: str = os.environ.get("REID_API_KEY", "")
    expose_paths: bool = _get_bool("REID_EXPOSE_PATHS", False)
    max_topk: int = int(os.environ.get("REID_MAX_TOPK", "50"))
    default_topk: int = int(os.environ.get("REID_DEFAULT_TOPK", "5"))
    max_upload_mb: int = int(os.environ.get("REID_MAX_UPLOAD_MB", "10"))
    score_threshold: Optional[float] = (
        float(os.environ["REID_SCORE_THRESHOLD"])
        if os.environ.get("REID_SCORE_THRESHOLD")
        else None
    )
