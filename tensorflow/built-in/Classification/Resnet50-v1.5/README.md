**Resnet50-v1.5 (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的ResNet50-v1.5网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-模型支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.默认参数说明](#3-默认参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.免责声明](#5-免责声明)
* [6.Release_Notes](#6-Release_Notes)


# 1. 模型概述
Resnet50-v1.5是原ResNet50-v1残差卷积网络的改进版，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。

ResNet50-v1.5网络结构的代码实现可参考：[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

| Models        | Framework   | Supported MLU | Supported Data Precision | Multi-GPUs | Multi-Nodes |
|---------------|-------------|---------------|--------------------------|------------|-------------|
| Resnet50-v1.5 | TensorFlow1 | MLU370-X8     | FP16/FP32                | Yes        | Not Tested  |

## 2.2 **推理模型支持情况**
| Models   | Framework   | Supported MLU | Supported Data Precision | Jit/Eager Support |
|----------|-------------|---------------|--------------------------|-------------------|
| Resnet50-v1.5 | TensorFlow1 | MLU370-X4/X8  | FP32                     | Eager             |

# 3. 默认参数说明

| 参数                 | 作用 | 默认值 |
|---------------------|------|------|
| mode                | 脚本的执行模式 | train_and_evaluate |
| batch_size          | 批大小 | 128 |
| lr_init             | 学习率的初始值 | 0.256 |
| iter_unit           | 迭代的单位 | epoch |
| lr_warmup_epochs    | 学习率预热次数 | 8 |
| warmup_steps        | 预热次数, 在性能测量中不被考虑在内 | 100 |
| weight_init         | 模型权重初始化方法 | fan_in |
| export_dir          | 导出的SavedModel的目录 | "./export_dir" |
| run_iter            | 单次运行的训练迭代次数 | -1 |
| label_smoothing     | 标签平滑度的值 | 0.1 |
| momentum=0.875      | 动量优化器的SGD动量值 | 0.875 |
| weight_decay        | 权重衰减比例系数 | 3.0517578125e-05 |
| data_format         | 数据格式 | NHWC |
| results_dir         | 写训练日志，总结和检查点的目录 | "." |
| use_dali            | 是否使用dali数据集 | False |
| model_dir           | 表示保存结果路径 | \ |
| finetune_checkpoint | 预先训练的检查点的路径，将用于微调 | "" |
| data_dir            | TFRecord格式的数据集的路径 | \ |
| data_idx_dir        | 数据集的路径 | \ |
| num_iter            | 要运行的迭代次数 | 90 |
| use_horovod         | 是否使用horovod分布式 | False |
| use_profiler        | 是否使用tfprof工具 | False |
| use_performance     | 是否进行性能测试 | False |
| use_amp             | 是否使用混合精度 | False |


# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成Resnet50-v1.5网络的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-X4/X8均可；
* Cambricon Driver >=v4.20.6；
* CNTensorFlow = 1.15.5;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 **环境准备**
### 4.2.1 **容器环境搭建**
容器环境通常有两种搭建方式，一种是基于基础镜像，另一种则是基于DOCKERFILE。

**(1)基于base docker image的容器环境搭建**

**a)导入镜像**

下载Cambricon TensorFlow1 docker镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow1_Image.tar.gz`

**b)启动容器**

`run_docker.sh`示例如下，根据本地的镜像版本，修改如下示例中的`IMAGE_NAME`和`IMAGE_TAG`变量后再运行`bash run_docker.sh`即可启动容器。
```bash
#!/bin/bash
# Below is a sample of run_docker.sh.
# Modify the  YOUR_IMAGE_NAME according to your own environment.
# For instance, IMAGE_NAME=tensorflow1-1.14.0-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=YOUR_IMAGE_TAG

export MY_CONTAINER="tf1_resnet50_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Classification/Resnet50-v1.5` 目录。

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
cd tensorflow_modelzoo/tensorflow/built-in/Classification/Resnet50-v1.5

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=resnet50_v1.5_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**

上一步成功运行后，本地便生成了一个名为`resnet50_v1.5_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为resnet50_v1.5_image
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**
该ResNet50-v1.5脚本基于ImageNet2012训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
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
完成数据集准备后，根据数据集实际路径修改`env.sh`内的`DATA_DIR`的值和`run_docker.sh`中需要拷贝进入docker容器内的数据集路径

如需使用dali数据集，请先生成dali_index：
```
bash dali_index.sh PATH/ilsvrc12_tfrecord PATH/dali_index/
备注：PATH/ilsvrc12_tfrecord是数据集tfrecord的路径，PATH/dali_index/是生成dali_index的路径。
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

| Models        | Framework   | MLU       | Data Precision | Cards | Run                                        |
|---------------|-------------|-----------|----------------|-------|--------------------------------------------|
| Resnet50-v1.5 | TensorFlow1 | MLU370-X8 | FP32           | 1     | bash ResNet50_Float32_90E_1MLU.sh          |
| Resnet50-v1.5 | TensorFlow1 | MLU370-X8 | FP32           | 8     | bash Horovod_ResNet50_Float32_90E_8MLUs.sh |
| Resnet50-v1.5 | TensorFlow1 | MLU370-X8 | AMP            | 8     | bash Horovod_ResNet50_AMP_90E_8MLUs.sh     |

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`epochs`，`np`等，按照上述命令即可开始from_scratch的分布式训练。

训练完成后，程序会输出训练精度`accuracy`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_ResNet50_Float32_90E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，假设加载resnet50的model.ckpt-805000进行finetune训练，训练1000 steps
# 首先把checkpoint拷贝到`PATH_TO_CKPT/..`目录下, `PATH_TO_CKPT`为model.ckpt-805000的绝对路径
# 再将iter_unit设置为batch，num_iter设置为1000, finetune_checkpoint设置为checkpoint路径`PATH_TO_CKPT`

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet50_model_${timestamp}"
checkpoint_dir=PATH_TO_CKPT

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 resnet50_trainer.py \
	--mode=train_and_evaluate \
	--batch_size=128 \
	--lr_init=0.256 \
	--iter_unit=batch \
	--lr_warmup_epochs=8 \
	--warmup_steps=100 \
	--weight_init=fan_in \
	--export_dir="./export_dir" \
	--run_iter=-1 \
	--label_smoothing=0.1 \
	--momentum=0.875 \
	--weight_decay=3.0517578125e-05 \
	--data_format=NHWC \
	--results_dir="."  \
	--use_dali=False \
	--model_dir=$model_dir \
	--finetune_checkpoint=$checkpoint_dir \
	--data_dir=$DATA_DIR \
	--data_idx_dir=$DATA_IDX_DIR \
	--num_iter=1000 \
	--use_horovod=True \
	--use_profiler=False \
	--use_performance=False \
	--use_amp=False
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


# 5. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6. Release_Notes
@TODO
