**Centernet (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的Centernet网络，支持训练。


------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.模型训练与推理参数说明](#3-模型训练与推理参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.结果展示](#5-结果展示)
  * [5.1训练结果](#51-训练结果)
* [6.免责声明](#6-免责声明) 
* [7.Release_Notes](#7-Release_Notes)


# 1. **模型概述**
Centernet是一个基于Anchor-free的目标检测算法，输入是图片，输出是带有目标检测框的图片。原始论文为[Object as Points](https://arxiv.org/pdf/1904.07850.pdf)

Centernet网络的TensorFlow原生代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/official/projects/centernet)。

# 2. **模型支持情况**

## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Centernet | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support|
|----- | ----- | ----- | ----- | ----- |
|Centernet   | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32   | Eager|


# 3. **模型训练与推理参数说明**

Centernet模型的训练与推理参数主要存在于`centernet.py`内，同时受到`mlu_centernet_config.config`及run_scripts/内的shell脚本的共同影响。

run_scripts/内的shell脚本涉及到的常用参数及含义如下表所示：

| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 训练的batch_size | 16   |
| num_train_steps | 不为0时，表示用户自主设定的每个epoch的步数。 | 0 |
| fine_tune_ckpt | 加载的checkpoint对象 | None|
| checkpoint_dir | checkpoint存储路径，供推理或finetune时使用 | None|
| do_train | 是否进行训练，若为False则表示进行推理 | True|
| use_amp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| use_gpu| 是否使用gpu进行训练 | False |
| use_profiler| 是否支持tensorboard，若为True则表示| False |
| use_performance | 是否开启性能测试，若为True则表示开启，训练结束后可在summary/summary.json内读出throughput与e2e| False |


# 4. **快速使用**

下面将详细展示如何在 Cambricon TensorFlow2上完成Centernet的训练和推理。

## 4.1 **依赖项检查**

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-S4/X4/X8均可；
* Cambricon Driver >=v4.20.6；
* CNTensorFlow >= 2.5.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 **环境准备**

### 4.2.1 **容器环境搭建**
容器环境通常有两种搭建方式，一种是基于基础镜像，另一种则是基于DOCKERFILE。

**(1)基于base docker image的容器环境搭建**

**a)导入镜像**

下载Cambricon TensorFlow2 docker镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow2_Image.tar.gz`

**b)启动容器**

`run_docker.sh`示例如下，根据本地的镜像版本，修改如下示例中的`IMAGE_NAME`变量后再运行`bash run_docker.sh`即可启动容器。
```bash
#!/bin/bash
# Below is a sample of run_docker.sh.
# Modify the  YOUR_DOCKER_IMAGE_NAME according to your own environment.
# For instance, IMAGE_NAME=tensorflow2-1.12.1-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_DOCKER_IMAGE_NAME
IMAGE_TAG=latest

export MY_CONTAINER="centernet_tensorflow_modelzoo"

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
```
**c)下载项目代码**

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Detection/centernet` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../../../tools/record_time/
pip install .

```
**(2)基于DOCKERFILE的容器环境搭建**

**a)构建镜像**

由于本仓库包含各类网络，如ASR类，NLP类，为避免网络之间可能的依赖项冲突，您可基于DOCKERFILE构建当前网络专属的镜像。详细步骤如下所示：
```bash
# 1. 新建并进入文件夹
mkdir dir_for_docker_build
cd dir_for_docker_build

# 2. 使用git clone下载tensorflow_modelzoo仓库

# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow2/built-in/Detection/centernet

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=your_docker_image_name
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**

上一步成功运行后，本地便根据您的命名生成了一个名为`your_docker_image_name`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为your_docker_image_name
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**
本仓库使用的是`COCO 2017`数据集。数据集下载：[https://cocodataset.org](https://cocodataset.org)
需要将数据集转换为tfrecord格式，可参见：[https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py)
在`run_scripts/`下运行`prepare_dataset.sh`(需要设置`DATASET`及`DATASET_LABEL`环境变量的值为本地对应数据集的路径)即可。
本地数据集目录结构需要与下方保持一致:
```bash
./data
└── coco2017_tfrecords
    ├── mscoco_label_map.pbtxt
    └── tfrecords
         ├── coco_train.record-00000-of-00100
         ├── coco_train.record-00001-of-00100
         ├── ...
         ├── coco_val.record-00008-of-00010
         └── coco_val.record-00009-of-00010
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
`run_scripts/`目录下提供了from_scratch的训练脚本。


Models | Framework | MLU | Data Precision | Cards | Run
----- | ----- | ----- | ----- | ----- | ----- |
Centernet | TensorFlow2 | MLU370-X8 | FP32 | 8 |bash Horovod_Centernet_Float32_8MLUs.sh
Centernet | TensorFlow2 | MLU370-X8 | AMP  | 8 |bash Horovod_Centernet_AMP_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`batch_size`，`steps`，`use_amp`等，进入`run_scripts`目录后，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_Centernet_Float32_8MLUs.sh
```

### 4.3.2 **一键执行推理脚本**
`run_scripts/`目录下还提供了推理脚本：`eval.sh`，您可根据自己的需求修改该脚本内的`checkpoint_dir`
```bash
bash eval.sh

```


# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

Centernet的训练精度由基于训练ckpt进行推理得到的mAP来表征。

Models   | MLUs |Batch Size  | Steps  |Precision(FP32)  | Precision(Mixed Precision)  |
----- | ----- | ----- | ----- | ----- | ----- |
Centernet  | 8 | 8 | 140000 | 29.8 | 24.8





# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO



