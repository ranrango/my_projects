#!/usr/bin/env python3
"""Run portfolio-level smoke checks for the tracked CV projects."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    cwd: Path
    command: list[str]


CHECKS = [
    Check(
        name="轻量 YOLOv8 检测脚本编译",
        cwd=ROOT / "基于YOLOv8的轻量化实时行人与车辆检测系统",
        command=[
            sys.executable,
            "-m",
            "py_compile",
            "detector.py",
            "run_camera.py",
            "run_video.py",
            "test_yolo.py",
            "onnx_out.py",
        ],
    ),
    Check(
        name="MOT+异常系统自检",
        cwd=ROOT / "多目标跟踪与异常行为识别系统（进阶）",
        command=[sys.executable, "validate_system.py"],
    ),
    Check(
        name="ReID 训练/部署脚本编译",
        cwd=ROOT / "reid属性识别系统",
        command=[
            sys.executable,
            "-m",
            "py_compile",
            "reid_baseline/train.py",
            "reid_baseline/market1501_loader.py",
            "reid_baseline/inference/evaluate_reid.py",
            "reid_baseline/deploy/api.py",
            "reid_baseline/deploy/build_gallery.py",
            "reid_baseline/deploy/search.py",
            "reid_baseline/deploy/export_torchscript.py",
            "reid_baseline/deploy/visualize_search.py",
        ],
    ),
    Check(
        name="端侧人脸门禁系统自检",
        cwd=ROOT / "端侧人脸识别门禁系统（工程+安全）",
        command=[sys.executable, "validate_system.py"],
    ),
    Check(
        name="仓库隐私扫描",
        cwd=ROOT,
        command=[sys.executable, "scripts/privacy_scan.py"],
    ),
]


def run_check(check: Check) -> bool:
    print(f"\n=== {check.name} ===")
    result = subprocess.run(
        check.command,
        cwd=check.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.rstrip())
    if result.returncode == 0:
        print(f"[PASS] {check.name}")
        return True
    print(f"[FAIL] {check.name}")
    return False


def main() -> int:
    passed = 0
    for check in CHECKS:
        if run_check(check):
            passed += 1
    total = len(CHECKS)
    print(f"\nPortfolio validation result: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
