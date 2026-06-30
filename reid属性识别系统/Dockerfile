FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REID_DEVICE=cpu \
    REID_CHECKPOINT=/app/checkpoints/reid_baseline.pth \
    REID_INDEX=/app/deploy/gallery_index.npz

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY reid_baseline/requirements.txt /app/reid_baseline/requirements.txt
RUN pip install --no-cache-dir -r /app/reid_baseline/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "reid_baseline.deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]

