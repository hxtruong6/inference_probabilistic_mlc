# Convenience targets. Install with: pip install -e ".[dev]"  (add ,image for X-ray)
DOCKER_IMAGE_NAME = dacaf_mlc
TIME = $(shell date +%Y%m%d%H%M%S)

docker-run:
	docker build -t $(DOCKER_IMAGE_NAME) .
	docker run -it $(DOCKER_IMAGE_NAME)
docker-exec:
	docker run -it $(DOCKER_IMAGE_NAME) /bin/bash
eval:
	python -m dacaf_mlc.evaluate
reproduce:
	bash scripts/reproduce_tabular.sh
test:
	python -m pytest tests/ -q
lint:
	ruff check dacaf_mlc tests
dataset_chest_xray:
	python dacaf_mlc/chest_xray_dataset/chest_xray_dataset.py
