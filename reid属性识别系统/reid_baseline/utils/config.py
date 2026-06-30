from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(config, overrides):
    merged = deepcopy(config)
    for key_path, value in overrides.items():
        if value is None:
            continue
        cursor = merged
        keys = key_path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return merged


def project_path(path):
    return Path(path).expanduser()
