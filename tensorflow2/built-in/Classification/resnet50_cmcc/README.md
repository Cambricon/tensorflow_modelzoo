**ResNet50 (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的ResNet50网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.模型支持情况](#2.支持情况)
* [3.默认参数配置](#3.默认参数配置)
  * [3.1模型训练默认参数配置](#3.1模型训练默认参数配置) 
* [4.快速使用](#4.快速使用)
  * [4.1依赖项检查](#4.1依赖项检查)
  * [4.2环境准备](#4.2环境准备)
  * [4.3运行Run脚本](#4.3运行Run脚本)
* [5.免责声明](#6.免责声明) 
* [6.Release notes](#7.Release_Notes)


# 1. 模型概述
ResNet50网络是残差卷积神经网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。

ResNet50网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50_CMCC | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**


# 3. 默认参数配置

## 3.1 **模型训练默认参数配置**

| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 更改训练的batch_size | 128 |
| model_dir | 指向保存checkpoint的路径 | ./mlu_model |
| data_dir | 指向数据集的路径 | \ |
| train_epochs | 更改训练的epoch数目 | 90 |
| base_learning_rate | 更改训练初始学习率 | 0.1 |
| use_profiler | 为True则开启tensorboard | False |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
  

  
# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成ResNet50的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
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
# Modify the  YOUR_IMAGE_NAME according to your own environment.
# For instance, IMAGE_NAME=tensorflow2-1.12.1-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=latest

export MY_CONTAINER="tensorflow_modelzoo"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER

if [ 0 -eq $num ];then
    xhost +
    docker run -it --name="${MY_CONTAINER}" \
     --net=host \
     --privileged=true \
     --cap-add=sys_ptrace \
     --shm-size="64g" \
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Classification/resnet50_cmcc` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../tools/record_time/
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
cd tensorflow_modelzoo/tensorflow2/built-in/Classification/resnet50_cmcc

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=resnet50_networks_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`resnet50_networks_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为resnet50_networks_image
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**
该ResNet50脚本基于ImageNet2012训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
需要将数据集转换为tfrecord格式，可参见：[https://github.com/tensorflow/models/tree/master/research/slim/datasets](https://github.com/tensorflow/models/tree/master/research/slim/datasets)
本地数据集目录结构请与下方保持一致：
``` bash
├── train-00000-of-01024
├── train-00001-of-01024
├── ...
├── validation-00000-of-00128
├── validation-00001-of-00128
├── ...
└── labels.txt
```

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow2  | MLU370-X8  | FP32  | 8  | Horovod_ResNet50_Float32_90E_8MLUs.sh
ResNet50  | TensorFlow2  | MLU370-X8  | AMP | 8  | Horovod_ResNet50_AMP_90E_8MLUs.sh
ResNet50  | TensorFlow2  | MLU370-X8  | FP32 | 1  | ResNet50_Float32_90E_1MLU.sh

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`train_steps`，`np`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
Horovod_ResNet50_Float32_90E_8MLUs.sh
```
训练完成后，程序会输出训练精度`accuracy`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_ResNet50_Float32_90E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载resnet50的ckpt-608652进行finetune训练，训练1000 step
# 则train_steps应设为609652，将model_dir修改为存放checkpoint的目录

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir=Your_MODEL_DIR
data_dir=YOUR_DATA_PATH

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 resnet_trainer.py \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --mode=train_and_eval \
    --distribution_strategy=off \
    --batch_size=128 \
    --steps_per_loop=312 \
    --train_epochs=90 \
    --use_synthetic_data=False \
    --use_performance=False \
    --use_amp=True \
    --use_horovod=True \
    --log_steps=1 \
    --run_eagerly=False \
    --enable_checkpoint_and_export=True \
    --base_learning_rate=0.1 \
    --train_steps=609652 \
    --use_profiler=False \
    --enable_tensorboard=False \
    --tf_mlu_thread_mode=mlu_private \
    --tf_gpu_thread_mode=gpu_private \
    --per_mlu_thread_count=2 \
    --per_gpu_thread_count=2 \
    --inter_op_parallelism_threads=4 \
    --intra_op_parallelism_threads=8 \
    --datasets_num_private_threads=2 \
    --epochs_between_evals=4 \
    --host_tracer_level=2 \
    --device_tracer_level=1 \
    --profiler_dir=$model_dir
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


# 5. **结果展示**

## 5.1 **训练结果**

** Training performance results: MLU370-X8**

在运行时传入`--use_performance=True --mode=train --train_steps=60 --steps_per_loop=1`参数。
以下性能结果基于cambricon-tensorflow2(v1.13.2)取得。下表中的Total Batch Size含义为所有MLU卡上的batch_size，例如当使用8卡MLU时，Total Batch Size为1024，因此每张卡的batch_size为1024/8=128。

Models   | MLUs   |  Total Batch Size  | Throughput(FP32)  | Throughput(Mixed Precision) 
----- | ----- | ----- | ----- | ----- 
ResNet50 | 1  |  128 | 196.83  | 383.29  
ResNet50 | 8  |  1024 |  1543.92 | 3006.60  


# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO



