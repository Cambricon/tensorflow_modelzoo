**Transformer_estimator(TensorFlow)**

本仓库是在MLU上基于TensorFlow框架实现的网络，支持训练与推理。


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
  * [5.2推理结果](#52-推理结果)
* [6.免责声明](#6-免责声明) 
* [7.Release_Notes](#7-Release_Notes)


# 1. 模型概述

Transformer_estimator 是一个基于 attention 机制的 NLP 模型。原始论文为[Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

Transformer_estimator 网络的TensorFlow原生代码实现可参考：[这里](https://github.com/tensorflow/models/tree/r1.13.0/official/transformer)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models                | Framework   | Supported MLU | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
--------------------- | ----------- | ------------- | ------------------------- | ----------- | ----------- |
Transformer_estimator | TensorFlow  | MLU370-X8     |          FP16/FP32| Yes         | Not Tested

## 2.2 **推理模型支持情况**


Models                | Framework   | Supported MLU   | Supported Data Precision |  Jit/Eager Support
--------------------- | ----------- | --------------- | ------------------------ | ------------------ |
Transformer_estimator | TensorFlow  | MLU370-X4/X8    | FP32                     | Eager



# 3. 模型训练推理参数说明

Transformer_estimator 模型的训练推理参数存在于`transformer_main.py`内，同时受到`run_scripts/`内的shell脚本的共同影响。

（1）`run_scripts`/内的shell脚本涉及到的常用参数及含义如下表所示：

| 参数                | 作用           | 默认值 |
|---------------------|----------------|------|
| mode                | 脚本的执行模式 | train |
| use_mlu             | 是否使用 MLU   | True |
| num_mlus            | 使用 MLU 个数  | 1 |
| num_gpus            | 使用 GPU 个数  | 0 |
| batch_size          | 批大小         | 4096  |
| model_dir           | 导出的模型的目录 | "./transformer_model_dir" |
| param_set           | 参数规模       | big |
| data_dir            | 数据集的路径   |  \  |
| vocab_file          | 字典文件的路径 |  \  |
| bleu_source         | 测试文件的路径 |  \  |
| bleu_ref            | 对比文件的路径 |  \  |
| train_steps         | 训练运行的步数 |  800000 |
| steps_between_evals | 评估间隔的步数 |  800000 |
| use_horovod         | 是否使用horovod分布式 | False |
| use_profiler        | 是否使用tfprof工具 | False |
| use_performance     | 是否进行性能测试 | False |
| use_amp             | 是否使用混合精度 | False |


# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow上完成Transformer_estimator的训练与推理。

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

export MY_CONTAINER="tf1_transformer_estimator_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/Transformer/transformer_estimator` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../../../../tools/record_time/
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
cd tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/Transformer/transformer_estimator

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=transformer_estimator_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../..

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`transformer_estimator_image`的镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为transformer_estimator_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
本仓库使用的训练数据集是[wmt14](https://www.statmt.org/wmt14/index.html)，依照如下文件列表下载：
```bash
newstest2014.de
newstest2014.en
vocab.ende.32768
wmt32k-dev-00001-of-00001*
wmt32k-train-00001-of-00100*
...
wmt32k-train-00100-of-00100*
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
进入`run_scripts/`，该目录内提供了from_scratch的训练脚本。


Models                | Framework  | Supported MLU | Data Precision | Cards | Run
--------------------- | ---------- | ------------- | -------------- | ----- | ----------------------------------------------- |
Transformer_estimator | TensorFlow | MLU370-X8     | Float32        |  8    | bash Horovod_Transformer_Float32_800000S_8MLUs.sh
Transformer_estimator | TensorFlow | MLU370-X8     | AMP            |  8    | bash Horovod_Transformer_AMP_800000S_8MLUs.sh
Transformer_estimator | TensorFlow | MLU370-X8     | Float32        |  1    | bash Transformer_Float32_800000S_1MLU.sh

根据您的实际环境与需求，修改脚本内数据集的路径（`env.sh`内的`DATA_DIR`）及其他参数的值，如`batch_size`,`use_amp`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_Transformer_Float32_800000S_8MLUs.sh
```

若您想基于已有的预训练模型进行训练，需要将已有的 ckpt 文件拷贝到 model_dir 中, 再执行下列命令：

```bash
# 以 float32 为例
# 可按需求适当调整 train_steps 训练步数
bash Horovod_Transformer_Float32_800000S_8MLUs.sh
```


### 4.3.2 **一键执行推理脚本**

进入`run_scripts/`，该目录内提供了单机单卡推理脚本：

推理脚本中的 batch_size 未显式提供，由 translate.py 中的 `_DECODE_BATCH_SIZE` 提供。

Models                | Framework  | Supported MLU | Data Precision  | Cards | Run 
--------------------- | ---------- | ------------- | --------------- | ----- | ---------------------------------------- | 
Transformer_estimator | TensorFlow | MLU370 X4/X8  | FP32            |  1    | bash Infer_Transformer_Float32_Bsz_32.sh

运行推理脚本之前，您需要将脚本内`model_dir`变量的值改为训练得到的checkpoint文件的实际路径。


# 5. **结果展示**

以下结果由镜像版本 tensorflow:v1.14.0-x86_64-ubuntu18.04-py3 得到。

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

Transformer_estimator 的训练精度可由模型在测试集上的`Bleu score`表征。在`newstest2014`数据集上，以fp32精度类型训练的模型精度如下：

MLUs | Batch Size(Train)/Batch Size(Test)  | STEPS | Bleu score(uncased) |Bleu score(cased)  
---- | ----------------------------------- | ----- |-------------------- | ---------------- |
8    | 4096/32                             | 100000 | 25                  | 24



## 5.2  **推理结果**

在单机单卡上使用MLU370-X8对训练了800000个STEPS的checkpoint进行推理，其精度与性能表现如下表所示:

Models                |  Jit/Eager   | Supported Data Precision  | Batch Size | Bleu score(uncased) |Bleu score(cased)
--------------------- | ------------ | ------------------------- | ---------- | ------------------- | ---------------- |
Transformer_estimator | Eager        | FP32                      | 32         | 25                  | 24



# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO
