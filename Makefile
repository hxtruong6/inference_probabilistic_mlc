# Write command to build docker and run it
DOCKER_IMAGE_NAME = inference_prob_mlc
docker-run:
	docker build -t $(DOCKER_IMAGE_NAME) .
	docker run -it $(DOCKER_IMAGE_NAME)
docker-exec:
	docker run -it $(DOCKER_IMAGE_NAME) /bin/bash