**DeepFM(TensorFlow)**

DeepFM是基于MLU的TensorFlow1框架实现的网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.模型训练推理参数说明](#3-模型训练推理参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1环境依赖项检查](#41-环境依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)  
* [5.结果展示](#5-结果展示)
  * [5.1训练结果](#51-训练结果)
* [6.免责声明](#6-免责声明) 
* [7.Release_Notes](#7-Release_Notes)


# 1. 模型概述

DeepFM 是Wide & Deep推荐模型的升级版，原始论文为[DeepFM](https://arxiv.org/abs/1703.04247)。

DeepFM 网络的TensorFlow1原生代码实现可参考：[这里](https://github.com/ChenglongChen/tensorflow-DeepFM)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models     | Framework   | Supported MLU | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
---------- | ----------- | ------------- | ------------------------- | ----------- | ----------- |
DeepFM     | TensorFlow  | MLU370-X8     |          FP16/FP32        | Yes         | Not Tested

## 2.2 **推理模型支持情况**


Models     | Framework   | Supported MLU   | Supported Data Precision |  Jit/Eager Support
---------- | ----------- | --------------- | ------------------------ | ------------------ |
DeepFM     | TensorFlow  | MLU370-X4/X8/S4 |          FP32            | Eager



# 3. 模型训练推理参数说明

DeepFM 模型训练推理的默认参数在`deepFM.py`文件中，参数也可通过`run_scripts`目录下的shell脚本传值。

（1）`run_scripts`内的shell脚本涉及到的常用参数及含义如下表所示：

| 参数                | 作用                  | 默认值         |
|---------------------|-----------------------|----------------|
| mode                | 脚本的执行模式        | train_and_eval |
| exec_mode           | 执行模型名            | DeepFM         |
| data_dir            | 数据集路径            | None           |
| batch_size          | 批数据的大小          | 4096           |
| use_gpu             | 是否使用gpu           | False          |
| skip_eval           | 是否跳过推理          | True           |
| learning_rate       | 学习率                |  0.001         |
| num_splits          | 拆分数量              |  none          |
| epoch               | epoch                 |  30            |
| use_horovod         | 是否使用horovod分布式 | False          |
| use_profiler        | 是否使用tfprof工具    | False          |
| use_performance     | 是否进行性能测试      | False          |
| use_amp             | 是否使用混合精度      | False          |


# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成DeepFM的训练推理。

## 4.1 **环境依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-X4/X8均可；
* Cambricon Driver >=v4.20.6；
* CNTensorFlow == 1.15.5;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 **环境准备**
### 4.2.1 **容器环境搭建**
容器环境通常有两种搭建方式，一种是基于基础镜像，另一种则是基于DOCKERFILE。

**(1)基于base docker image的容器环境搭建**

**a)导入镜像**  

下载Cambricon TensorFlow 镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow1_Image.tar.gz`

**b)启动容器**  

`run_docker.sh`示例如下，根据本地的镜像版本，修改如下示例中的`IMAGE_NAME`和`IMAGE_TAG`变量后再运行`bash run_docker.sh`即可启动容器。

```bash
#!/bin/bash
# Below is a sample of run_docker.sh.
# Modify the  YOUR_IMAGE_NAME according to your own environment.
# For instance, 
# IMAGE_NAME=tensorflow1-1.15.0-x86_64-ubuntu18.04
# IMAGE_TAG=latest

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=YOUR_IMAGE_TAG

export MY_CONTAINER="tf1_deepfm_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Recommendation/DeepFM` 目录。

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

由于本仓库包含各类网络，如 Segmentation 类，Classification 类，为避免网络之间可能的依赖项冲突，您可基于DOCKERFILE构建当前网络专属的镜像。详细步骤如下所示：
```bash
# 1. 新建并进入文件夹
mkdir dir_for_docker_build
cd dir_for_docker_build

# 2. 使用git clone下载tensorflow_modelzoo仓库
git clone https://gitee.com/cambricon/tensorflow_modelzoo.git

# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow/built-in/Recommendation/DeepFM/

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=deepfm_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../..

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`deepfm_image`的镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为deepfm_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
本仓库使用的训练数据集是[Porto Seguro's Safe Driver Prediction competition on Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data)，下载解压后如下所示：
```bash
sample_submission.csv
train.csv
test.csv
```

## 4.3 **运行Run脚本**

进入`run_scripts`，该目录内提供了from_scratch的训练脚本，该网络没有专属推理脚本。

Models | Framework  | Supported MLU | Data Precision | Cards | Run
-------| ---------- | ------------- | -------------- | ----- | ----------------------------------------------- |
DeepFM | TensorFlow | MLU370-X8     | Float32        |  8    | bash Horovod_DeepFM_Float32_30E_8MLUs.sh
DeepFM | TensorFlow | MLU370-X8     | AMP            |  8    | bash Horovod_DeepFM_AMP_30E_8MLUs.sh
DeepFM | TensorFlow | MLU370-X8     | Float32        |  1    | bash DeepFM_Float32_30E_1MLU.sh

根据您的实际环境与需求，修改脚本内数据集的路径（`env.sh`内的`DATA_DIR`）及其他参数值，如`batch_size`,`use_amp`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_DeepFM_Float32_30E_8MLUs.sh
```

# 5. **结果展示**

以下结果由镜像版本 tensorflow:v1.14.0-x86_64-ubuntu18.04-py3 得到。

**Training accuracy results: MLU370-X8**

DeepFM 在 `Porto Seguro's Safe Driver Prediction competition on Kaggle` 数据集上，以fp32精度类型训练的模型精度如下：

   MLUs   | Batch Size  |   EPOCH  |  accuracy    
--------- | ------------| -------- |------------ |
    8     |    1024     |    30    |  0.266   

# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO
