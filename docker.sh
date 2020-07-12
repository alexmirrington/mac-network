#!/bin/sh

# Get script directory
THIS_DIR="$(cd "$(dirname "$0")"; pwd)";

# Docker image information.
URL=$(git -C $THIS_DIR remote get-url origin )
TAG=$(git -C $THIS_DIR rev-parse --short HEAD)
REPO=$(basename -s .git $URL)
USER=$(basename $(dirname $URL))
NAME="$USER/$REPO"
IMG="$NAME:$TAG"

# Dataset directories TODO take from args
DATA_DIR="$HOME/documents/hdd/arch/datasets/gqa"
GLOVE_DIR="$HOME/documents/hdd/arch/embeddings/glove"

# Build container if it does not exist for this commit
if [ ! "$(docker image ls -q -f reference=$IMG)" ]; then
    docker build -t $IMG $THIS_DIR
fi

# Run container, cleaning up if a stopped container with the same name
# already exists.
if [ ! "$(docker ps -q -f name=$NAME)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$NAME)" ]; then
        docker rm $NAME
    fi
    BASEDIR="/usr/src/"
    docker run -it --gpus all \
    -u $(id -u):$(id -g) \
    -w $BASEDIR \
    --mount type=bind,source="$THIS_DIR",target=$BASEDIR \
    --mount type=bind,source="$THIS_DIR/data",target="$BASEDIR/data" \
    --mount type=bind,source="$GLOVE_DIR",target="$BASEDIR/data/glove",readonly \
    --mount type=bind,source="$DATA_DIR/images/images",target="$BASEDIR/data/images",readonly \
    --mount type=bind,source="$DATA_DIR/images/spatial",target="$BASEDIR/data/spatial",readonly \
    --mount type=bind,source="$DATA_DIR/images/objects",target="$BASEDIR/data/objects",readonly \
    --mount type=bind,source="$DATA_DIR/questions",target="$BASEDIR/data/questions",readonly \
	$IMG bash
fi