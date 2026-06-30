import json
import os


def load_config(path: str, base_dir: str = "") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if base_dir:
        _resolve_paths(cfg, base_dir)
    return cfg


def _resolve_paths(cfg: dict, base_dir: str):
    path_keys = {"log_file", "snapshot_dir", "index_file", "faces_dir"}
    for k, v in cfg.items():
        if isinstance(v, dict):
            _resolve_paths(v, base_dir)
        elif isinstance(v, str) and k in path_keys and not os.path.isabs(v):
            cfg[k] = os.path.join(base_dir, v)
