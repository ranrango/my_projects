# Portfolio Delivery Design

## Goal

Turn the four tracked computer-vision projects into a GitHub-ready portfolio package for interviewers and technical reviewers. The repository should be easy to scan in five minutes, runnable in fifteen minutes, and credible enough to discuss implementation details in an interview.

## Scope

This design covers only the projects already tracked in the GitHub repository:

- `基于YOLOv8的轻量化实时行人与车辆检测系统`
- `多目标跟踪与异常行为识别系统（进阶）`
- `reid属性识别系统`
- `端侧人脸识别门禁系统（工程+安全）`

Untracked local folders such as learning notes, experiments, datasets, raw videos, model weights, and personal files remain out of scope unless explicitly added later.

## Delivery Shape

The repository will be organized as a portfolio-first package:

- Root `README.md`: concise project map, comparison table, visual result section, quick validation commands, and links to deeper project docs.
- `DELIVERY.md`: a reviewer-oriented guide with 5-minute, 15-minute, and 30-minute review paths.
- `assets/<project>/`: lightweight public screenshots, GIFs, thumbnails, and result charts. No private faces, raw surveillance footage, datasets, or model weights.
- `scripts/validate_all.py`: one command to run smoke checks for all tracked projects and summarize results.
- Project READMEs: consistent sections for goal, demo, quick start, metrics, architecture, deliverables, limitations, and upgrade path.
- Optional `resources/README.md`: instructions for where users should place external model weights, sample videos, and datasets.

## Per-Project Presentation

### Lightweight YOLOv8 Detection

Show this as the entry-level edge detection project. Highlight CPU-friendly ONNX/OpenCV inference, video/camera entry points, and clean detector abstraction. Add a screenshot or GIF of person/vehicle detection, a benchmark table, and explicit model placement instructions.

### MOT and Anomaly Detection

Show this as the strongest PoC-style engineering project. Highlight ByteTrack, ROI filtering, JSONL event output, alarm snapshots, and the existing industrial delivery kit. Add result images from the industrial kit, event format, validation command, and a boundary note that real deployment requires field labels and model acceleration.

### ReID System

Show this as the most complete model-to-service pipeline. Highlight training, evaluation, gallery indexing, FastAPI serving, Docker deployment, and TorchScript export. Add the Market1501 metric table already present in docs and a small search-result screenshot or auto-generated chart if no image is available.

### Edge Face-Gate System

Show this as a full application workflow: enrollment, recognition, liveness check, access decision, lockout, and audit logs. Use anonymous or synthetic visuals only. State clearly that the current LBP/Haar implementation is a CPU engineering baseline and that production accuracy should use ArcFace or MobileFaceNet ONNX.

## Resource Policy

Large and sensitive assets stay out of Git:

- Exclude model weights, raw videos, private face images, datasets, audit logs, `.env` files, and local machine paths.
- Use `assets/` only for small public images and GIFs.
- Document external files with expected path, purpose, and generation/download instructions.
- If a required resource is missing, smoke checks should skip that deep demo gracefully and report a clear message.

## Validation

The portfolio is considered ready when:

- `python3 scripts/validate_all.py` runs and reports per-project status.
- Existing self-checks still pass:
  - Face-gate validation: 11/11 checks.
  - MOT validation: system validation passes.
  - Detection scripts compile.
  - ReID scripts compile.
- A privacy scan finds no hardcoded API-key-shaped secrets, local user paths, or demo credentials in tracked files.
- Root README and project READMEs can be understood without downloading private data.

## Non-Goals

- Do not ship real personal face images or private surveillance data.
- Do not add large model weights or datasets to Git.
- Do not redesign the algorithms or retrain models unless resources are explicitly provided.
- Do not turn this into a production monitoring platform; this is a portfolio and PoC presentation layer.

## Open Resource Inputs

The implementation should work with existing tracked files, but the final visual polish improves if the user provides:

- One public sample video or output video for detection.
- One public sample video or output video for MOT/anomaly.
- One ReID search-result image or permission to generate a toy-result visualization.
- One synthetic or anonymized face-gate screenshot.
- Final metric values to show in the root comparison table.
