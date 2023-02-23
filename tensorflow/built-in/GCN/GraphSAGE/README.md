**GraphSAGE (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的GraphSAGE网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.默认参数说明](#3-默认参数说明)
  * [3.1训练参数说明](#31-训练参数说明)
  * [3.2推理参数说明](#32-推理参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.免责声明](#5-免责声明)
* [6.Release_Notes](#6-Release_Notes)


# 1. 模型概述
这个目录包含运行GraphSage算法所需的代码。GraphSage可以被看作是图形卷积的随机泛化，它对包含丰富特征信息的大规模动态图形特别有用。关于该算法的细节，请参见我们的[论文](https://arxiv.org/pdf/1706.02216.pdf)。

# 2. 模型支持情况

## 2.1 **训练模型支持情况**
| Models    | Framework   | Supported MLU | Supported Data Precision | Multi-GPUs | Multi-Nodes |
|-----------|-------------|---------------|--------------------------|------------|-------------|
| GraphSage | TensorFlow1 | MLU370-X8     | FP16/FP32                | Yes        | Not Tested  | 

## 2.2 **推理模型支持情况**
| Models    | Framework   | Supported MLU   | Supported Data Precision | Jit/Eager Support |
|-----------|-------------|-----------------|--------------------------|-------------------|
| GraphSage | TensorFlow1 | MLU370-X4/X8/S4 | FP32                     | Eager             |

# 3. 默认参数说明

## 3.1 **训练参数说明**
| 参数              | 作用            | 默认值                                  |
|-----------------|---------------|--------------------------------------|
| base_log_dir    | 训练后模型和日志的保存目录 | "${work_dir}/GraphSAGE_model_single" |
| data_dir        | 数据集路径         | "${DATA_DIR}/ppi"                    |
| epochs          | 训练的epoch数     | 10                                   |
| batch_size      | 训练的批大小        | 512                                  |
| max_total_steps | 每个epoch个训练步数  | 1000000000                           |
| use_amp         | 是否使用混合精度训练    | False                                |
| use_profiler    | 是否使用tfperf工具  | False                                |
| use_performance | 是否进行性能测试      | False                                |

## 3.2 **推理参数说明**
| 参数        | 作用          | 默认值         |
|-----------|-------------|-------------|
| data_dir  | 数据集路径       | ${DATA_DIR} |
| embed_dir | ckpt存放路径    | 0           |
| setting   | 设置val还是test | test        |

# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成GraphSage的训练与推理。

## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-X4/X8/S4均可；
* Cambricon Driver >=v4.20.6；
* CNTensorFlow == 1.15.5;
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
# Modify the YOUR_IMAGE_NAME and IMAGE_TAG according to your own environment.
# For instance,
# IMAGE_NAME=tensorflow1-1.14.0-x86_64-ubuntu18.04
# IMAGE_TAG=YOUR_IMAGE_TAG

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=YOUR_IMAGE_TAG

export MY_CONTAINER="tensorflow_graphsage_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/GCN/GraphSAGE` 目录。

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
git clone https://gitee.com/cambricon/tensorflow_modelzoo.git

# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow/built-in/GCN/GraphSAGE

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=graphsage_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../
```

**b)创建并启动容器**

上一步成功运行后，本地便生成了一个名为`graphsage_image`的镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为graphsage_image
# 2. 运行run_docker.sh
bash run_docker.sh
```

### 4.2.2 **数据集准备**
本仓库使用的训练数据集是`Protein-Protein Interactions`数据集，可从[此处](http://snap.stanford.edu/graphsage/ppi.zip)下载。下载至本地并解压后，数据集的存放路径可参考下方的目录结构：
```bash
/data/tensorflow/training/datasets/GraphSage_dataset/ppi
├──ppi-class_map.json
├──ppi-feats.npy
├──ppi-G.json
├──ppi-id_map.json
└──ppi-walks.txt
```

### 4.2.4 **环境变量修改**

完成上述准备后，根据数据集实际路径修改`env.sh`内的`DATA_DIR`的值。

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

进入`run_scripts/`，该目录内提供了用于from_scratch的训练脚本。

| Models    | Framework   | Supported MLU | Data Precision | Cards | Run                                |
|-----------|-------------|---------------|----------------|-------|------------------------------------|
| GraphSAGE | TensorFlow1 | MLU370-X4/X8  | FP32           | 1     | bash GraphSAGE_Float32_10E_1MLU.sh |
| GraphSAGE | TensorFlow1 | MLU370-X4/X8  | AMP            | 1     | bash GraphSAGE_AMP_10E_1MLU.sh     |

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`, `max_total_steps`等，按照上述命令即可开始from_scratch训练。

### 4.3.2 **一键执行推理脚本**
本仓库提供了GraphSAGE网络的推理脚本：`run_scripts/infer_run_eager_GraphSAGE.sh`，需要用户手动传入模型参数，默认以`tensorflow_modelzoo/tensorflow/built-in/GCN/GraphSAGE`为当前目录。具体参见`3. 默认参数配置`，具体示例如下：
```bash
cd run_scripts
bash infer_run_eager_GraphSAGE.sh PATH_TO_CKPT
```

# 5.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6.Release_Notes
@TODO
