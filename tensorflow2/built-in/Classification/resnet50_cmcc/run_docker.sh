#/bin/bash

export MY_CONTAINER="modelzoo_tensorflow"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
xhost +
docker run -it --shm-size '64gb' --privileged --network=host --cap-add=sys_ptrace --device=/dev/cambricon_dev0 --device=/dev/cambricon_ipcm0 --device=/dev/cambricon_ctl -v /data/:/data/  yellow.hub.cambricon.com/tensorflow/tensorflow2:v1.13.0-x86_64-ubuntu18.04 /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
