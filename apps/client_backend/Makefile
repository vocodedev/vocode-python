

dep-local:
	rm -rf $(HOME)/.local/lib/python3.11/site-packages/vocode
	ln -s $(PWD)/../../vocode $(HOME)/.local/lib/python3.11/site-packages/vocode

backend:
	uvicorn --host 0.0.0.0 --port 8000 --env-file .env --reload main:app


IMAGE_NAME=vocode-client-backend
IMAGE_VERSION=0.0.1

DOCKER_REGISTRY=yantao0527
DOCKER_CLIENT_BACKEND=$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_VERSION)

docker-build:
	docker build -t $(DOCKER_CLIENT_BACKEND) .

docker-run:
	docker run -d \
		--name client-backend \
		--env-file=.env \
		-p 3000:3000 \
		$(DOCKER_CLIENT_BACKEND)

LOCATION=us-central1
PROJECT_ID=ai-agents-386717
REPOSITORY_NAME=cloud-run-source-deploy
REGISTRY_PREFIX=$(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)
GCLOUD_CLIENT_BACKEND=$(REGISTRY_PREFIX)/$(IMAGE_NAME):$(IMAGE_VERSION)

gcloud-push:
	docker tag $(DOCKER_VOCODE_DEMO) $(GCLOUD_CLIENT_BACKEND)
	docker push $(GCLOUD_CLIENT_BACKEND)