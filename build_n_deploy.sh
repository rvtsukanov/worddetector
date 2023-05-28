#!/usr/bin/env bash

IAM_TOKEN=$(yc iam create-token)
CONTAINER_REGISTRY_URL='cr.yandex'
REGISTRY_NAME='docker-football'

TAG='latest'

docker login \
  --username iam \
  --password $IAM_TOKEN \
  $CONTAINER_REGISTRY_URL

REGISTRY_ID=$(yc container registry get $REGISTRY_NAME | head -n 1 | cut -d' ' -f2)

echo $REGISTRY_ID

IMAGE_NAME=$CONTAINER_REGISTRY_URL/$REGISTRY_ID/tesser:$TAG

docker build . -t $IMAGE_NAME
docker push $IMAGE_NAME
