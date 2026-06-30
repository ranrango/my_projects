.PHONY: install smoke build-gallery api docker-build docker-up

install:
	python -m pip install -r reid_baseline/requirements.txt

smoke:
	python -m py_compile reid_baseline/deploy/*.py reid_baseline/*.py

build-gallery:
	python -m reid_baseline.deploy.build_gallery \
		--checkpoint checkpoints/reid_baseline.pth \
		--gallery-dir gallery \
		--output deploy/gallery_index.npz \
		--device auto

api:
	uvicorn reid_baseline.deploy.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

