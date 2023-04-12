**Common_networks (TensorFlow2)**

本仓库是在MLU上基于TensorFlow框架实现的网络，共支持ResNet50、ResNet101、DenseNet201、Vgg19四种模型，支持训练与推理。

------------

**目录 (Table of Contents)**
- [1. 模型概述](#1-模型概述)
- [2. 模型支持情况](#2-模型支持情况)
  - [2.1 训练模型支持情况](#21-训练模型支持情况)
  - [2.2 推理模型支持情况](#22-推理模型支持情况)
- [3. 默认参数配置](#3-默认参数配置)
  - [3.1 模型训练默认参数配置](#31-模型训练默认参数配置)
  - [3.2 模型推理默认参数说明](#32-模型推理默认参数说明)
- [4.快速使用](#4-快速使用)
  - [4.1 依赖项检查](#41-依赖项检查)
  - [4.2 环境准备](#42-环境准备)
  - [4.3 运行Run脚本](#43-运行Run脚本)
- [5.结果展示](#5-结果展示)
- [6.免责声明](#6-免责声明)
- [7.Release_Notes](#7-Release_Notes)


# 1. 模型概述


ResNet50和ResNet101网络都是残差卷积网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。
ResNet网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/model/legacy/image_classification/resnet)。

DenseNet201网络是密集连接卷积网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1608.06993.pdf)。
DenseNet201网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)。

Vgg19网络的原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)。
Vgg19网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
ResNet101 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
DenseNet201 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
Vgg19 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


## 2.2 **推理模型支持情况**

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support|
|----- | ----- | ----- | ----- | ----- |
|ResNet50   | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32   | Eager|
|ResNet101   | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32   | Eager|
|Vgg19   | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32   | Eager|
|Densenet201   | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32   | Eager|



# 3. 默认参数说明

## 3.1 **模型训练与推理参数说明**

常用参数均在`classifier.py`内，详细作用如下：

| 参数 | 作用 |
|------|------|
| batch_size | 更改训练的batch_size |
| model_dir | 指向保存checkpoint的路径 |
| data_dir | 指向数据集的路径 |
| epochs | 更改训练的epoch数目 |
| use_amp | 控制是否使用amp进行混合精度训练或验证 |
| skip_eval | 是否跳过推理阶段 |
| finetune_checkpoint | 预训练模型的路径。若进行推理时，此参数指向用于推理的已训练好的checkpoint文件 |
| enable_tensorboard | 控制是否开启tensorboard，并记录性能 |
| distribution_strategy | 控制是否开启原生分布式，原生分布式不能与Horovod分布式同时开启 |
| num_mlus，num_gpus | 联合控制网络运行的设备，在mlu设备上运行需设置num_mlus=1,num_gpus=0；在gpu设备上运行需设置num_mlus=0,num_gpus=1 |



# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成分类网络的训练与推理。
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
IMAGE_TAG=YOUR_DOCKER_IMAGE_TAG

export MY_CONTAINER="classification_common_network_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Classification/common_networks` 目录。

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
cd tensorflow_modelzoo/tensorflow2/built-in/Classification/common_networks

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
此demo基于ImageNet2012训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
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

完成数据集准备后，根据数据集实际路径修改`env.sh`内的`DATA_DIR`的值。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 1  | bash ResNet50_Float32_90E_1MLU.sh
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 8  | bash Horovod_ResNet50_Float32_90E_8MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | AMP  | 8  | bash Horovod_ResNet50_AMP_90E_8MLUs.sh
ResNet101  | TensorFlow  | MLU370-X8  | FP32  | 1  | bash ResNet101_Float32_100E_1MLU.sh
ResNet101  | TensorFlow  | MLU370-X8  | FP32  | 8  | bash Horovod_ResNet101_Float32_100E_8MLUs.sh
ResNet101  | TensorFlow  | MLU370-X8  | AMP | 8  | bash Horovod_ResNet101_AMP_100E_8MLUs.sh
Vgg19  | TensorFlow  | MLU370-X8  | FP32  | 1  | bash Vgg19_Float32_100E_1MLU.sh
Vgg19  | TensorFlow  | MLU370-X8  | FP32  | 8  | bash Horovod_Vgg19_Float32_100E_8MLUs.sh
Vgg19  | TensorFlow  | MLU370-X8  | AMP  | 8  | bash Horovod_Vgg19_AMP_100E_8MLUs.sh
DenseNet201  | TensorFlow  | MLU370-X8  | FP32  | 1  | bash DenseNet201_Float32_140E_1MLU.sh
DenseNet201  | TensorFlow  | MLU370-X8  | FP32  | 8  | bash Horovod_DenseNet201_Float32_140E_8MLUs.sh
DenseNet201  | TensorFlow  | MLU370-X8  | AMP | 8  | bash Horovod_DenseNet201_AMP_140E_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`train_steps`，`np`等，进入`run_scripts`目录下，按照上述命令即可开始from_scratch的分布式训练。

训练完成后，程序会输出训练精度`accuracy`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_DenseNet201_Float32_140E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载已经训练了50个epoch的checkpoint文件进行finetune，训练1000 step
# 则finetune_steps应设为1000，epochs应设为51

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/densenet201_model_${timestamp}"
data_dir=YOUR_DATA_PATH
ckpt_file=YOUR_PATH/mlu_model

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 classifier.py \
    --dataset=imagenet \
    --model_type=densenet201 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=64 \
    --epochs=51 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=True \
    --run_eagerly=False \
    --skip_eval=False \
    --finetune_steps=1000 \
    --finetune_checkpoint=$ckpt_file \
    --enable_tensorboard=False \
    --datasets_num_private_threads=0
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **一键执行推理脚本**

本仓库提供了常用分类网络的推理脚本：`run_scripts/Infer_${network_name}_*.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`use_amp`，并根据您的本地实际路径，修改`../env.sh`内相关的预训练模型(如`VGG19_CKPT`)路径。完成修改后，按照如下命令运行即可分别以不同的参数推理。

目前支持的精度类型与推理模式组合以及运行环境如下所示：

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support| RUN |
|----- | ----- | ----- | ----- | ----- | ----- |
|ResNet50   | TensorFlow2  | MLU370-S4/X4/X8  | FP32   | Eager| bash Infer_ResNet50_Eager_Float32_Bsz_128.sh |
|ResNet50   | TensorFlow2  | MLU370-S4/X4/X8  | FP16   | Eager| bash Infer_ResNet50_Eager_AMP_Bsz_128.sh |
|ResNet101   | TensorFlow2  | MLU370-S4/X4/X8  | FP32   | Eager| bash Infer_ResNet101_Eager_Float32_Bsz_128.sh |
|ResNet101   | TensorFlow2  | MLU370-S4/X4/X8  | FP16   | Eager| bash Infer_ResNet101_Eager_AMP_Bsz_128.sh |
|Vgg19   | TensorFlow2  | MLU370-S4/X4/X8  | FP32   | Eager| bash Infer_Vgg19_Eager_Float32_Bsz_128.sh |
|Vgg19   | TensorFlow2  | MLU370-S4/X4/X8  | FP16   | Eager| bash Infer_Vgg19_Eager_AMP_Bsz_128.sh |
|Densenet201   | TensorFlow2  | MLU370-S4/X4/X8  | FP32   | Eager| bash Infer_Densenet201_Eager_Float32_Bsz_64.sh |
|Densenet201   | TensorFlow2  | MLU370-S4/X4/X8  | FP16   | Eager| bash Infer_Densenet201_Eager_AMP_Bsz_64.sh |


# 5.结果展示

**Training accuracy results: MLU370-X8**

图像分类任务的训练精度通常用`top1`表征，在本仓库中，最终的训练精度由`accuracy`表征。最终的训练精度如下所示：

Models  | MLUs |  Mixed Precision Top1   | FP32 Top1
----- | ----- | ----- | ----- |
ResNet50 | 8  | 0.7542 | 0.7540
ResNet101 | 8  | 0.7591 | 0.7662
DenseNet201 | 8  | 0.6978 | 0.6972
Vgg19 | 8  | 0.6934 | 0.6957


# 6. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. Release_Notes
@TODO




