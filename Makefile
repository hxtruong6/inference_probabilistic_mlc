# Write command to build docker and run it
DOCKER_IMAGE_NAME = inference_prob_mlc
TIME = $(shell date +%Y%m%d%H%M%S)
docker-run:
	docker build -t $(DOCKER_IMAGE_NAME) .
	docker run -it $(DOCKER_IMAGE_NAME)
docker-exec:
	docker run -it $(DOCKER_IMAGE_NAME) /bin/bash
eval:
	python inference_evaluate_models.py
dataset_chest_xray:
	python src/chest_xray_dataset.py