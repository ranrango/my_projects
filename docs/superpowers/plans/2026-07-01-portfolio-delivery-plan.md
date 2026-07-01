# 作品集交付版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前 GitHub 已跟踪的四个计算机视觉项目整理成面向面试官的作品集交付版。

**Architecture:** 交付层只增加文档、轻量可视化资产、统一验收脚本和资源说明，不改核心算法路径。仓库根目录负责 5 分钟总览，每个项目 README 负责 15 分钟技术深读，脚本负责 30 分钟可运行验证。

**Tech Stack:** Markdown、Python 3 标准库、已有项目的 OpenCV/NumPy/PyTorch 依赖、SVG 轻量图像资产。

---

## 文件结构

本计划只触碰 GitHub 已跟踪项目和新增交付层文件：

- 创建 `DELIVERY.md`：面试官阅读路径和交付清单。
- 创建 `resources/README.md`：模型、视频、数据集、隐私数据的放置说明。
- 创建 `scripts/privacy_scan.py`：仓库内隐私和密钥形态扫描。
- 创建 `scripts/validate_all.py`：四个项目的统一验收入口。
- 创建 `scripts/build_portfolio_assets.py`：生成可公开提交的轻量 SVG 图表。
- 创建 `assets/README.md` 和 `assets/*/*.svg`：根 README 和项目 README 引用的展示图。
- 修改 `README.md`：增加视觉结果、指标总览、快速验收、资源策略。
- 修改四个项目 README：统一增加作品集展示结构、结果指标、边界说明。

不触碰未跟踪目录，例如 `langchain/`、`提示词工程/`、`drone-object-detection/`、`my-uv-project/`。

### Task 1: 新增隐私扫描脚本

**Files:**
- Create: `scripts/privacy_scan.py`

- [ ] **Step 1: 先运行缺失脚本，确认红灯**

Run:

```bash
python3 scripts/privacy_scan.py
```

Expected: 命令失败，并提示 `scripts/privacy_scan.py` 不存在。

- [ ] **Step 2: 创建 `scripts/privacy_scan.py`**

写入以下代码：

```python
#!/usr/bin/env python3
"""Scan tracked portfolio files for publish-blocking private data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "api-key-shaped secret": re.compile(r"sk-proj-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{12,}"),
    "hardcoded api_key assignment": re.compile(
        r"api_key\\s*=\\s*[\\\"']sk-[A-Za-z0-9_-]{12,}[\\\"']"
    ),
    "local user path": re.compile(r"/Users/randemac|file:///Users/randemac"),
    "mac temp path": re.compile(r"/private/var/folders/|/var/folders/"),
    "demo rtsp password": re.compile(r"rtsp://user:password"),
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".sh",
    ".env",
    ".example",
    ".svg",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in output.splitlines() if line.strip()]


def is_text_candidate(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name == ".env.example"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    hits: list[tuple[str, str]] = []
    for path in tracked_files():
        if not is_text_candidate(path) or not path.exists():
            continue
        text = read_text(path)
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                hits.append((str(path), label))

    if hits:
        print("Privacy scan failed:")
        for path, label in hits:
            print(f"- {path}: {label}")
        return 1

    print("Privacy scan passed: no targeted hits in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 验证隐私扫描脚本**

Run:

```bash
python3 scripts/privacy_scan.py
```

Expected:

```text
Privacy scan passed: no targeted hits in tracked files.
```

- [ ] **Step 4: 提交**

```bash
git add scripts/privacy_scan.py
git commit -m "chore: add portfolio privacy scan"
```

### Task 2: 新增统一验收脚本

**Files:**
- Create: `scripts/validate_all.py`

- [ ] **Step 1: 先运行缺失脚本，确认红灯**

Run:

```bash
python3 scripts/validate_all.py
```

Expected: 命令失败，并提示 `scripts/validate_all.py` 不存在。

- [ ] **Step 2: 创建 `scripts/validate_all.py`**

写入以下代码：

```python
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
    print(f"\\n=== {check.name} ===")
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
    print(f"\\nPortfolio validation result: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 验证统一验收脚本**

Run:

```bash
python3 scripts/validate_all.py
```

Expected: 末尾输出：

```text
Portfolio validation result: 5/5 checks passed
```

- [ ] **Step 4: 提交**

```bash
git add scripts/validate_all.py
git commit -m "chore: add portfolio validation runner"
```

### Task 3: 生成作品集轻量视觉资产

**Files:**
- Create: `scripts/build_portfolio_assets.py`
- Create: `assets/README.md`
- Create: `assets/detection/overview.svg`
- Create: `assets/mot-anomaly/overview.svg`
- Create: `assets/reid/metrics.svg`
- Create: `assets/face-gate/workflow.svg`

- [ ] **Step 1: 先运行缺失脚本，确认红灯**

Run:

```bash
python3 scripts/build_portfolio_assets.py
```

Expected: 命令失败，并提示 `scripts/build_portfolio_assets.py` 不存在。

- [ ] **Step 2: 创建 `scripts/build_portfolio_assets.py`**

写入以下代码：

```python
#!/usr/bin/env python3
"""Generate small public SVG assets for the portfolio README pages."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\\n", encoding="utf-8")


def card(title: str, subtitle: str, items: list[str], accent: str) -> str:
    rows = []
    y = 112
    for item in items:
        rows.append(f'<text x="36" y="{y}" class="item">• {item}</text>')
        y += 34
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="{title}">
  <style>
    .bg {{ fill: #f7f9fb; }}
    .panel {{ fill: #ffffff; stroke: #d9e2ec; stroke-width: 2; }}
    .title {{ font: 700 32px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }}
    .subtitle {{ font: 500 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526173; }}
    .item {{ font: 500 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #243447; }}
    .accent {{ fill: {accent}; }}
  </style>
  <rect class="bg" width="960" height="420" rx="0"/>
  <rect class="panel" x="24" y="24" width="912" height="372" rx="8"/>
  <rect class="accent" x="24" y="24" width="12" height="372" rx="6"/>
  <text x="60" y="70" class="title">{title}</text>
  <text x="60" y="100" class="subtitle">{subtitle}</text>
  {''.join(rows)}
</svg>
"""


def reid_metrics() -> str:
    bars = [
        ("mAP", 0.5588, "#2f80ed"),
        ("Rank-1", 0.7957, "#16a34a"),
        ("Rank-5", 0.9121, "#f59e0b"),
        ("Rank-10", 0.9362, "#dc2626"),
    ]
    rows = []
    y = 112
    for label, value, color in bars:
        width = int(value * 620)
        rows.append(f'<text x="70" y="{y + 23}" class="label">{label}</text>')
        rows.append(f'<rect x="190" y="{y}" width="620" height="28" rx="4" fill="#e5e7eb"/>')
        rows.append(f'<rect x="190" y="{y}" width="{width}" height="28" rx="4" fill="{color}"/>')
        rows.append(f'<text x="830" y="{y + 23}" class="value">{value:.4f}</text>')
        y += 58
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="ReID metrics">
  <style>
    .bg {{ fill: #f7f9fb; }}
    .panel {{ fill: #ffffff; stroke: #d9e2ec; stroke-width: 2; }}
    .title {{ font: 700 32px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }}
    .subtitle {{ font: 500 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526173; }}
    .label {{ font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #243447; }}
    .value {{ font: 600 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #243447; }}
  </style>
  <rect class="bg" width="960" height="420"/>
  <rect class="panel" x="24" y="24" width="912" height="372" rx="8"/>
  <text x="60" y="70" class="title">ReID Market1501 评估结果</text>
  <text x="60" y="100" class="subtitle">10 轮训练记录，适合作品集展示和面试讨论</text>
  {''.join(rows)}
</svg>
"""


def main() -> None:
    write(
        ASSETS / "detection" / "overview.svg",
        card(
            "轻量 YOLOv8 检测",
            "YOLOv8n ONNX + OpenCV DNN，面向边缘 CPU 推理",
            ["支持 person / car / bus / truck", "视频、摄像头、无头批处理入口", "本地样例验证：640x360 视频可跑通"],
            "#2f80ed",
        ),
    )
    write(
        ASSETS / "mot-anomaly" / "overview.svg",
        card(
            "多目标跟踪与异常识别",
            "ByteTrack + ROI + 规则引擎 + JSONL 事件留证",
            ["自带 validate_system.py 交付验收", "工业 PoC 套件包含配置、对比报告、结果样例", "边界：上线前需要现场标注和推理加速"],
            "#16a34a",
        ),
    )
    write(ASSETS / "reid" / "metrics.svg", reid_metrics())
    write(
        ASSETS / "face-gate" / "workflow.svg",
        card(
            "端侧人脸门禁系统",
            "注册、识别、活体、安全锁定、审计日志的完整闭环",
            ["全链路自检 11/11 通过", "隐私目录 enrolled_faces 与 audit_logs 不入库", "边界：生产精度建议升级 ArcFace/MobileFaceNet"],
            "#7c3aed",
        ),
    )
    write(
        ASSETS / "README.md",
        "# Portfolio Assets\\n\\n"
        "本目录只存放可公开提交的轻量展示图和结果图。不要提交真实人脸、原始监控视频、数据集、模型权重或访问日志。\\n",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行资产生成脚本**

Run:

```bash
python3 scripts/build_portfolio_assets.py
```

Expected: 生成 `assets/detection/overview.svg`、`assets/mot-anomaly/overview.svg`、`assets/reid/metrics.svg`、`assets/face-gate/workflow.svg`。

- [ ] **Step 4: 验证 SVG 文件存在**

Run:

```bash
find assets -type f | sort
```

Expected:

```text
assets/README.md
assets/detection/overview.svg
assets/face-gate/workflow.svg
assets/mot-anomaly/overview.svg
assets/reid/metrics.svg
```

- [ ] **Step 5: 提交**

```bash
git add scripts/build_portfolio_assets.py assets
git commit -m "docs: add portfolio visual assets"
```

### Task 4: 新增面试官交付索引

**Files:**
- Create: `DELIVERY.md`
- Create: `resources/README.md`

- [ ] **Step 1: 创建 `DELIVERY.md`**

写入以下内容：

```markdown
# 作品集交付说明

这个仓库包含四个计算机视觉工程项目，覆盖目标检测、多目标跟踪、行人重识别和端侧门禁。它面向作品集展示、面试讲解和 PoC 交流，不直接携带私有数据、模型权重或真实人脸。

## 5 分钟快速浏览

1. 看根目录 `README.md` 的项目矩阵和成果图。
2. 看 `assets/` 下的四张展示图。
3. 重点关注 ReID 的指标表和 MOT 的工业交付套件说明。

## 15 分钟技术深读

1. 运行 `python3 scripts/validate_all.py` 查看基础可运行性。
2. 阅读每个项目 README 的「技术亮点」「结果指标」「边界说明」。
3. 选择一个项目进入源码：检测看 `detector.py`，MOT 看 `core/bytetracker.py`，ReID 看 `reid_baseline/train.py`，门禁看 `face_gate/security/access_controller.py`。

## 30 分钟演示路径

1. 按 `resources/README.md` 放置模型、视频或数据。
2. 检测项目运行 `python run_video.py --video test.mp4 --no-display`。
3. MOT 项目运行 `python validate_system.py` 和 `python run_video.py --config config/default_config.json --no-display --max-frames 30`。
4. ReID 项目运行 toy 数据链路，验证训练、评估和检索服务入口。
5. 门禁项目运行 `python validate_system.py`，检查注册、检索、安全和审计闭环。

## 交付边界

- 不提交模型权重、数据集、真实门禁照片、RTSP 账号、访问日志和云服务密钥。
- 当前检测和门禁项目偏 CPU 工程基线，适合展示系统设计和端侧部署思路。
- 工业上线前需要现场数据标注、真实指标评估、推理加速、日志轮转和人工复核流程。
```

- [ ] **Step 2: 创建 `resources/README.md`**

写入以下内容：

```markdown
# 外部资源说明

本目录用于说明运行演示所需的外部资源。不要把大文件或隐私数据提交到 Git。

## 模型权重

- 检测项目：将 `yolov8n.onnx` 放到对应项目根目录，或按 README 使用 `onnx_out.py` 导出。
- MOT 项目：将 `yolov8n.onnx` 放到 `多目标跟踪与异常行为识别系统（进阶）/models/`。
- ReID 项目：将 `reid_baseline.pth` 放到 `reid属性识别系统/checkpoints/`。
- 门禁项目：当前基线不需要深度学习模型；生产升级可替换为 ArcFace 或 MobileFaceNet ONNX。

## 样例视频和数据

- 检测视频放到对应项目根目录，命名为 `test.mp4` 或运行时通过 `--video` 指定。
- MOT 视频放到对应项目根目录，命名为 `test.mp4` 或在配置文件中指定。
- ReID 数据推荐使用 Market1501；无真实数据时可运行 toy 数据生成脚本。
- 门禁人脸图片只放本地 `enrolled_faces/`，不要提交到仓库。

## 隐私规则

- 不提交真实人脸、监控原视频、访问日志、训练数据、API Key、RTSP URL 或本机绝对路径。
- 可公开展示的截图和 SVG 放入 `assets/`。
```

- [ ] **Step 3: 提交**

```bash
git add DELIVERY.md resources/README.md
git commit -m "docs: add portfolio delivery guide"
```

### Task 5: 重写根 README 的作品集展示层

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 插入顶部成果区**

在标题和项目目录之间加入以下 Markdown：

```markdown
## 作品集亮点

| 项目 | 展示重点 | 可运行验证 | 适合讨论的问题 |
|---|---|---|---|
| 轻量 YOLOv8 检测 | ONNX/OpenCV CPU 推理 | `python run_video.py --video test.mp4 --no-display` | 预处理、NMS、边缘部署 |
| MOT+异常识别 | ByteTrack、ROI、事件 JSONL | `python validate_system.py` | 跟踪匹配、报警规则、工业 PoC |
| ReID 属性识别 | 训练、评估、图库检索、API | toy Market1501 链路 | Triplet Loss、mAP/Rank-k、服务化 |
| 端侧人脸门禁 | 注册、活体、安全、审计闭环 | `python validate_system.py` | 端侧安全、隐私保护、升级路线 |

## 可视化结果

![轻量检测](assets/detection/overview.svg)
![MOT 与异常识别](assets/mot-anomaly/overview.svg)
![ReID 指标](assets/reid/metrics.svg)
![端侧门禁流程](assets/face-gate/workflow.svg)

## 一键验收

```bash
python3 scripts/validate_all.py
```

验收脚本会执行四个项目的 smoke checks，并额外扫描已跟踪文件中的密钥形态字符串、本机路径和示例口令。
```

- [ ] **Step 2: 更新说明区**

把 README 末尾说明区更新为：

```markdown
## 资源与隐私

- 模型权重、原始视频、数据集和访问日志不放入 Git。
- 可公开展示的小图、SVG 和结果图放入 `assets/`。
- 外部资源放置方式见 [resources/README.md](resources/README.md)。
- 面试官阅读路径见 [DELIVERY.md](DELIVERY.md)。
- 文档中的身份、摄像头地址和 API Key 均使用示例值；请勿提交真实人员姓名、门禁照片、RTSP 账号、访问日志、云服务密钥或本机绝对路径。
```

- [ ] **Step 3: 验证 README 引用的文件存在**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in [
    "assets/detection/overview.svg",
    "assets/mot-anomaly/overview.svg",
    "assets/reid/metrics.svg",
    "assets/face-gate/workflow.svg",
    "DELIVERY.md",
    "resources/README.md",
]:
    assert Path(path).exists(), path
print("README references exist")
PY
```

Expected:

```text
README references exist
```

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs: refresh root portfolio README"
```

### Task 6: 统一四个项目 README 的作品集段落

**Files:**
- Modify: `基于YOLOv8的轻量化实时行人与车辆检测系统/README.md`
- Modify: `多目标跟踪与异常行为识别系统（进阶）/README.md`
- Modify: `reid属性识别系统/README.md`
- Modify: `端侧人脸识别门禁系统（工程+安全）/README.md`

- [ ] **Step 1: 检测项目 README 增加作品集区块**

在功能特性后加入：

```markdown
## 作品集展示

![轻量检测概览](../assets/detection/overview.svg)

| 维度 | 内容 |
|---|---|
| 技术定位 | 边缘 CPU 目标检测基线 |
| 输入 | 摄像头、本地视频、单张图片 |
| 输出 | 检测框、类别、置信度、可选结果视频 |
| 验收 | 脚本编译通过；本地提供 `test.mp4` 和 `yolov8n.onnx` 后可运行视频检测 |
| 边界 | 仓库不携带模型权重和视频；真实场景上线需现场数据评估 |
```

- [ ] **Step 2: MOT README 增加作品集区块**

在“已具备能力”后加入：

```markdown
## 作品集展示

![MOT 与异常识别概览](../assets/mot-anomaly/overview.svg)

| 维度 | 内容 |
|---|---|
| 技术定位 | 工业场景 PoC：检测、跟踪、规则报警、事件留证 |
| 输入 | 视频、摄像头、RTSP |
| 输出 | 结构化 JSONL 事件、报警截图、标注视频 |
| 验收 | `python3 validate_system.py` |
| 边界 | 通用 COCO 权重不等于工业专用模型；上线前需要现场标注、精度评估和推理加速 |
```

- [ ] **Step 3: ReID README 增加作品集区块**

在项目结构前加入：

```markdown
## 作品集展示

![ReID 指标](../assets/reid/metrics.svg)

| 维度 | 内容 |
|---|---|
| 技术定位 | 行人重识别训练、评估、图库检索与 API 服务 |
| 指标记录 | Market1501 10 轮训练：mAP 0.5588，Rank-1 0.7957，Rank-5 0.9121，Rank-10 0.9362 |
| 部署链路 | checkpoint -> gallery index -> CLI search -> FastAPI -> Docker |
| 验收 | Python 脚本编译；toy Market1501 可跑通训练和评估链路 |
| 边界 | toy 数据只验证链路；真实部署需业务摄像头数据重新定阈值 |
```

- [ ] **Step 4: 门禁 README 增加作品集区块**

在系统架构前加入：

```markdown
## 作品集展示

![端侧门禁流程](../assets/face-gate/workflow.svg)

| 维度 | 内容 |
|---|---|
| 技术定位 | 端侧身份识别、安全控制和审计闭环示范 |
| 流程 | 人脸注册 -> 特征提取 -> 活体检测 -> 授权/拒绝 -> JSONL 审计 |
| 验收 | `python validate_system.py`，当前自检 11/11 通过 |
| 隐私 | `enrolled_faces/` 和 `audit_logs/` 不入库 |
| 边界 | Haar+LBP 是 CPU 工程基线；生产识别率建议升级 ArcFace/MobileFaceNet ONNX |
```

- [ ] **Step 5: 验证 Markdown 中的 assets 链接**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
checks = {
    "基于YOLOv8的轻量化实时行人与车辆检测系统/README.md": "../assets/detection/overview.svg",
    "多目标跟踪与异常行为识别系统（进阶）/README.md": "../assets/mot-anomaly/overview.svg",
    "reid属性识别系统/README.md": "../assets/reid/metrics.svg",
    "端侧人脸识别门禁系统（工程+安全）/README.md": "../assets/face-gate/workflow.svg",
}
for readme, rel in checks.items():
    text = Path(readme).read_text(encoding="utf-8")
    assert rel in text, readme
print("project README asset links exist")
PY
```

Expected:

```text
project README asset links exist
```

- [ ] **Step 6: 提交**

```bash
git add '基于YOLOv8的轻量化实时行人与车辆检测系统/README.md' '多目标跟踪与异常行为识别系统（进阶）/README.md' 'reid属性识别系统/README.md' '端侧人脸识别门禁系统（工程+安全）/README.md'
git commit -m "docs: standardize project portfolio sections"
```

### Task 7: 最终验证与推送

**Files:**
- No new files.

- [ ] **Step 1: 运行统一验收**

Run:

```bash
python3 scripts/validate_all.py
```

Expected:

```text
Portfolio validation result: 5/5 checks passed
```

- [ ] **Step 2: 检查 Git 状态**

Run:

```bash
git status --short --branch
```

Expected: 只剩用户未跟踪的本地杂项；没有本计划修改产生的未提交 tracked 文件。

- [ ] **Step 3: 推送**

Run:

```bash
git push github main
```

Expected:

```text
main -> main
```

## 自检记录

- 设计文档要求的根 README、交付索引、资源策略、统一验收脚本、项目 README、轻量资产均有对应任务。
- 计划不要求提交模型权重、真实人脸、原始视频、数据集、访问日志或密钥。
- 所有新增脚本只使用 Python 3 标准库。
- 验证命令覆盖隐私扫描、项目自检、脚本编译和资产链接。
