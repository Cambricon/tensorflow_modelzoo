**BERT_CRF (TensorFlow)**

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
BERT_CRF 是一种Transformer的双向编码器的 NLP 模型，原始论文为[BERT_CRF](https://arxiv.org/abs/1810.04805).

BERT_CRF网络的TensorFlow原生代码实现可参考：[这里](https://github.com/kyzhouhzau/BERT-NER)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
BERT_CRF | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**


Models  | Framework  | Supported MLU   | Supported Data Precision  |  Jit/Eager Support
----- | ----- | ----- | -----  | ----- |
BERT_CRF | TensorFlow  | MLU370-S4/X4/X8  | FP32 | Eager



# 3. 模型训练推理参数说明


BERT_CRF模型的训练推理参数存在于`BERT_CRF.py`内，同时受到`run_scripts/`内的shell脚本的共同影响。

（1）`run_scripts`/内的shell脚本涉及到的常用参数及含义如下表所示：



| 参数 | 作用 | 默认值 |
|------|------|------|
| task_name | 训练任务名称 | None |
| do_lower_case | 输入文本是否小写 | True |
| crf | 使用crf | True |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| use_amp | 是否使用amp模式 | False |
| finetune_steps | 控制finetune的步数 | 0 |
| hvd_device | 用来训练的horovod 设备 | None |
| max_seq_length | 标记化后的最大总输入序列长度 | 128 |
| num_train_epochs | 训练的 epoch 的次数 | 3 |
| save_checkpoints_steps| 保存模型权重的步数 | 1000 |
| do_train |  是否训练 | False |
| do_eval |  是否评估 | False |
| do_predict |  是否预测 | False |
| learning_rate | 学习率 | 5e-5 |
| vocab_file | 训练 BERT 模型的词汇文件 | None |
| bert_config_file | BERT 模型的配置文件 | None |
| init_checkpoint | 初始化权重 | None |
| train_batch_size | 更改训练的batch_size | None |
| data_dir | 输入数据所在文件夹 | None |
| output_dir | 保存模型文件的输出文件夹 | None |
| use_performance | 是否使用 use_performance 测试工具 | False |
| use_profiler | 是否使用 profiler 工具 | False |


# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow上完成BERT_CRF的训练与推理。
## 4.1 **环境依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNTensorFlow == 1.15.5;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 **环境准备**
### 4.2.1 **容器环境搭建**
容器环境通常有两种搭建方式，一种是基于基础镜像，另一种则是基于DOCKERFILE。

**(1)基于base docker image的容器环境搭建**

**a)导入镜像**  

下载Cambricon TensorFlow 镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow_Image.tar.gz`

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

export MY_CONTAINER="bert_crf_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/bert/bert_crf` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../../../../tools/record_time
pip install .

```

**(2)基于DOCKERFILE的容器环境搭建**

**a)构建镜像**  

由于本仓库包含各类网络，如Segmentation类，Classification 类，为避免网络之间可能的依赖项冲突，您可基于DOCKERFILE构建当前网络专属的镜像。详细步骤如下所示：
```bash
# 1. 新建并进入文件夹
mkdir dir_for_docker_build
cd dir_for_docker_build

# 2. 使用git clone下载tensorflow_modelzoo仓库
git clone https://gitee.com/cambricon/tensorflow_modelzoo.git


# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/bert/bert_crf

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=bert_crf_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`bert_crf_image`的镜像，后续即可基于该镜像创建容器。
```bash/
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为bert_crf_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
本仓库使用的训练数据集是[CoNLL-2003](https://github.com/kyzhouhzau/BERT-NER/tree/0f77e478872453df51cd3c65d1a39b12d9617f9d/data)，依照如下文件列表下载：
```bash
dev.txt
test.txt
train.txt
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
进入`run_scripts/`，该目录内提供了from_scratch的训练脚本。


Models  | Framework  | Supported MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
BERT_CRF| TensorFlow  | MLU370-X8  | Float32  |  8 |bash Horovod_BERT_CRF_Float32_16E_8MLUs.sh
BERT_CRF  | TensorFlow  | MLU370-X8  | AMP | 8  |bash Horovod_BERT_CRF_AMP_16E_8MLUs.sh
BERT_CRF  | TensorFlow  | MLU370-X8  | Float32 | 1 |bash BERT_CRF_Float32_4E_1MLU.sh

根据您的实际环境与需求，修改脚本内数据集的路径（env.sh内的DATA_DIR）及其他参数的值。

### 4.3.2 **一键执行推理脚本**
进入`run_scripts/`，该目录内提供了单机单卡推理脚本：

Models  | Framework  | Supported MLU   | Data Precision  | Cards|Run 
----- | ----- | ----- | ----- | ----- |  ----- | 
BERT_CRF  | TensorFlow  | MLU370 S4/X4/X8 | FP32  |  1 |bash Infer_BERT_CRF_Float32.sh



# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

BERT_CRF的训练精度可由模型在测试集上的`accuracy`表征。在`CoNLL-2003`数据集上，以fp32精度类型训练的模型精度如下：

MLUs | Batch Size(Train)/Batch Size(Test)  | EPOCH  |accuracy 
----- | ----- | ----- |-----  | 
8  | 32/32  | 16|0.9812



## 5.2  **推理结果**

在单机单卡(MLU370-X8)环境下对训练了16个EPOCH的checkpoint进行推理，其精度与性能表现如下表所示:

Models |  Jit/Eager   | Supported Data Precision  | Batch Size | accuracy 
----- | ----- | ----- | ----- | ----- |
BERT_CRF | Eager   | FP32  | 8  | 0.9812  



# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO



