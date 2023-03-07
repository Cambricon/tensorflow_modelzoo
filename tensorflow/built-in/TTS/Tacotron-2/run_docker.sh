#!/bin/bash
# Below is a sample of run_docker.sh.
# Modify the  IMAGE_NAME and IMAGE_TAG according to your own environment.
# For instance, IMAGE_NAME=tensorflow-1.12.1-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_DOCKER_IMAGE_NAME
IMAGE_TAG=YOUR_DOCKER_IMAGE_TAG

export MY_CONTAINER="tf1_tacotron2_tensorflow_modelzoo"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER

if [ 0 -eq $num ];then
    xhost +
    docker run -it --name="${MY_CONTAINER}" \
     --net=host \
     --privileged=true \
     --cap-add=sys_ptrace \
     --shm-size="16g" \
     -v /usr/bin/cnmon:/usr/bin/cnmon \
     -v /data:/data \
     --device=/dev/cambricon_dev0 \
     --device=/dev/cambricon_ctl \
     $IMAGE_NAME:$IMAGE_TAG  \
     /bin/bash
else
    docker start $MY_CONTAINER
    docker exec -ti --env COLUMNS=`tput cols` --env LINES=`tput lines` $MY_CONTAINER /bin/bash

fi
