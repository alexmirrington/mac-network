.PHONY: build run

REMOTE := $(shell git remote get-url origin)
REPO   := $(basename $(notdir $(REMOTE)))
USER   := $(notdir $(patsubst %/,%,$(dir $(REMOTE))))
TAG    := $(shell git rev-parse HEAD)
IMG    := ${USER}/${REPO}:${TAG}


run: build
	@docker run -u $(shell id -u):$(shell id -g) --gpus all -it $(IMG) bash

build:
	@docker build -t $(IMG) .
