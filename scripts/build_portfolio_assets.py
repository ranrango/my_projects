#!/usr/bin/env python3
"""Generate small public SVG assets for the portfolio README pages."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def card(title: str, subtitle: str, items: list[str], accent: str) -> str:
    rows = []
    y = 142
    for item in items:
        safe_item = escape(item)
        rows.append(
            f'  <rect class="itemRow" x="72" y="{y}" width="816" height="52" rx="8"/>\n'
            f'  <circle class="accentFill" cx="98" cy="{y + 26}" r="5"/>\n'
            f'  <text x="120" y="{y + 33}" class="item">{safe_item}</text>\n'
        )
        y += 64
    safe_title = escape(title)
    safe_subtitle = escape(subtitle)
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="{safe_title}">
  <style>
    .bg {{ fill: #f7f9fb; }}
    .panel {{ fill: #ffffff; stroke: #d9e2ec; stroke-width: 2; }}
    .title {{ font: 700 30px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }}
    .subtitle {{ font: 500 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526173; }}
    .item {{ font: 600 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #243447; }}
    .itemRow {{ fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1; }}
    .accent {{ fill: {accent}; }}
    .accentFill {{ fill: {accent}; }}
  </style>
  <rect class="bg" width="960" height="420" rx="0"/>
  <rect class="panel" x="28" y="28" width="904" height="364" rx="8"/>
  <rect class="accent" x="28" y="28" width="10" height="364" rx="5"/>
  <text x="72" y="76" class="title">{safe_title}</text>
  <text x="72" y="110" class="subtitle">{safe_subtitle}</text>
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
    y = 136
    for label, value, color in bars:
        width = int(value * 430)
        safe_label = escape(f"{label} · {value:.4f}")
        rows.append(
            f'  <text x="74" y="{y + 24}" class="label">{safe_label}</text>\n'
            f'  <rect x="300" y="{y + 5}" width="430" height="24" rx="5" fill="#e5e7eb"/>\n'
            f'  <rect x="300" y="{y + 5}" width="{width}" height="24" rx="5" fill="{color}"/>\n'
        )
        y += 58
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="ReID metrics">
  <style>
    .bg {{ fill: #f7f9fb; }}
    .panel {{ fill: #ffffff; stroke: #d9e2ec; stroke-width: 2; }}
    .title {{ font: 700 30px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }}
    .subtitle {{ font: 500 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526173; }}
    .label {{ font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #243447; }}
  </style>
  <rect class="bg" width="960" height="420"/>
  <rect class="panel" x="28" y="28" width="904" height="364" rx="8"/>
  <text x="72" y="76" class="title">ReID Market1501 评估结果</text>
  <text x="72" y="110" class="subtitle">10 轮训练记录，适合作品集展示和面试讨论</text>
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
            ["自带 validate_system.py 交付验收", "工业 PoC：配置、对比报告、结果样例", "边界：上线前需要现场标注与推理加速"],
            "#16a34a",
        ),
    )
    write(ASSETS / "reid" / "metrics.svg", reid_metrics())
    write(
        ASSETS / "face-gate" / "workflow.svg",
        card(
            "端侧人脸门禁系统",
            "注册、识别、活体、安全锁定、审计日志的完整闭环",
            ["全链路自检 11/11 通过", "隐私目录：enrolled_faces / audit_logs 不入库", "边界：生产精度建议升级 ArcFace/MobileFaceNet"],
            "#7c3aed",
        ),
    )
    write(
        ASSETS / "README.md",
        "# Portfolio Assets\n\n"
        "本目录只存放可公开提交的轻量展示图和结果图。不要提交真实人脸、原始监控视频、数据集、模型权重或访问日志。\n",
    )


if __name__ == "__main__":
    main()
