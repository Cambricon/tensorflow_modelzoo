**UNet_3D_Medical (TensorFlow)**

本仓库是在MLU上基于TensorFlow框架实现的 UNet_3D_Medical 网络，支持训练与推理。


------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.默认参数说明](#3-默认参数说明)
  * [3.1模型训练参数说明](#31-模型训练参数说明)
  * [3.2模型推理参数说明](#32-模型推理参数说明)
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
UNet 3D 是一个 3D 图像分割网络模型，输入是 3D 图像，输出是分割结果。原始论文为[3D U-Net](https://arxiv.org/pdf/1606.06650.pdf).

UNet 3D 网络的 NVIDIA 代码实现可参考：[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
------- | ---------- | --------------- | ------------------------- | ----------- | ----------- |
UNet 3D | TensorFlow | MLU370-X8       | FP16/FP32                 | Yes         | Not Tested

## 2.2 **推理模型支持情况**


Models  | Framework  | Supported MLU   | Supported Data Precision  |  Jit/Eager Support
------- | ---------- | --------------- | ------------------------  | ------------------ |
UNet 3D | TensorFlow | MLU370-X4/X8 | FP32                 | Eager


# 3. 默认参数说明

## 3.1 **模型训练参数说明**

UNet 3D 模型的训练参数存在于`unet3d.py`内，同时受到`run_scripts/`内的shell脚本的共同影响。

（1）`run_scripts`/内的shell脚本涉及到的常用参数及含义如下表所示：



| 参数 | 作用 | 默认值 |
|------|------|------|
| exec_mode | 选择执行模式。| train |
| batch_size | 训练的batch_size | 1   |
| data_dir | 训练数据集文件的路径| None |
| model_dir | 模型输出文件的路径。| None |
| max_steps | 最大的训练步数。 | 16000 |
| use_amp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | False |


  
## 3.2 **模型推理参数说明**

### 3.2.1 **模型推理常用参数说明**

UNet 3D 模型的推理与训练使用相同的脚本，主要区别在于 exec_mode 的不同。
UNet 3D 模型的推理参数存在于`unet3d.py`内，同时受到`run_scripts/`内的shell脚本的共同影响。

| 参数 | 作用 | 举例 |
|------|------|------|
| exec_mode | 选择执行模式, 推理时需要修改为 evaluate。|  evaluate |
| batch_size | 推理时使用的batch_size | 1   |
| data_dir | 训练数据集文件的路径| None |
| model_dir | 模型输出文件的路径。| None |
| use_amp | 控制是否进行混合精度训练 | True |
| steps | 训练的step数 | 1000 |
| log_every | 控制每次打印log的step间隔 | 100 |
| use_gpu | 设置为True则使用gpu,设置为False则使用mlu | False |
| benchmark | 启用原生代码性能测试 | False |
| warmup_steps | 开启benchmark参数后使用，设置跳过性能测试的step数 | 200 |
| fold | 训练K折交叉验证,范围在0-(K-1) | 0 |
| num_folds | 训练K折交叉验证,K的次数 | 5 |
| argument | 是否使用数据增强 | True |
| resume_training | 是否使用checkpoint恢复训练 | True |

# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow上完成 UNet 3D 的训练与推理。
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

下载Cambricon TensorFlow1 镜像并参考如下命令加载镜像：
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

export MY_CONTAINER="tf1_unet3d_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Segmentation/UNet_3D_Medical` 目录。

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
cd tensorflow_modelzoo/tensorflow/built-in/Segmentation/UNet_3D_Medical

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow1:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=unet3d_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`unet3d_image`的镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为unet3d_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
本仓库使用的训练数据集是[Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/)，需要遵循官网指引注册下载数据集, 最后处理成 tfrecord：

仓库中 models/dataset/preprocess_data.py 脚本可将原始数据转换成 tfrecord 格式供训练和推理使用。

```
python models/dataset/preprocess_data.py -i /data/<name/of/the/raw/data/folder> -o /data/<name/of/the/preprocessed/data/folder> -v
```
需保证与如下目录结构一致：
```bash
MICCAI_BraTS_2019_Data_Training_Preprocess
├── volume-0.tfrecord*
├── volume-1.tfrecord*
├── volume-10.tfrecord*
├── volume-11.tfrecord*
├── ...
├── volume-82.tfrecord*
├── volume-83.tfrecord*
└── volume-9.tfrecord*
```

随后，还需根据`MICCAI_BraTS_2019_Data_Training_Preprocess`的本地路径修改`env.sh`内的`DATA_DIR`的值。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
进入`run_scripts/`，该目录内提供了from_scratch的训练脚本。


Models   | Framework  | Supported MLU | Data Precision  | Cards  | Run
-------  | ---------- | ------------- | --------------- | ------ | ----- |
UNet 3D  | TensorFlow | MLU370-X8     | Float32         |  8     |bash Horovod_UNet_3D_Float32_16000S_8MLUs.sh
UNet 3D  | TensorFlow | MLU370-X8     | AMP             |  8     |bash Horovod_UNet_3D_AMP_16000S_8MLUs.sh
UNet 3D  | TensorFlow | MLU370-X8     | Float32         |  1     |bash UNet_3D_Medical_1000S_1MLU.sh


根据您的实际环境与需求，修改脚本内数据集的路径（`env.sh`内的`DATA_DIR`）及其他参数的值，如`batch_size`，`max_steps`，`use_amp`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_UNet_3D_Float32_16000S_8MLUs.sh
```
若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_UNet_3D_Float32_16000S_8MLUs.sh `为例）, 增加 resume_training 参数：

```bash
# fp32精度下，使用8卡MLU370-X8，
# 加载 model_dir 下的checkpoint文件进行finetune:
# finetune 1000代, 由于网络代数=max_steps/card_num，finetune1000代需要手动输入8000
horovodrun -np 8 python unet3d.py \
    --exec_mode=train \
    --data_dir=YOUR_DATA_SET_PATH \
    --max_steps=8000 \
    --use_horovod=True \
    --resume_training=True  \
    --model_dir=YOUR_MODEL_PATH
```

### 4.3.2 **一键执行推理脚本**
进入`run_scripts/`，该目录内提供了单机单卡推理脚本：

Models  | Framework  | Supported MLU   | Data Precision  | Cards |  Run 
------- | ---------- | --------------- | --------------- | ----- |  ----- | 
UNet3D  | TensorFlow  | MLU370 X4/X8 | FP32  |   1|bash Infer_UNet_3D_Medical_Float32_Bsz_2.sh


运行推理脚本之前，您需要将脚本内`model_dir`变量的值改为训练得到的checkpoint文件的实际路径。




# 5. **结果展示**

以下结果由镜像版本 tensorflow:v1.14.0-x86_64-ubuntu18.04-py3 得到。

## 5.1 **训练结果**

**Training performance results: MLU370-X8**

UNet3D fp32精度类型训练的模型性能如下：

 MLUs |Batch Size(Train)  | throughput_train [img/s] |  latency_train_mean [ms]
 ---- | ----------------- | ----------------- | -----------------------
 8    |  1                | 6.049411939064266 |  1328.4551461537678



## 5.2  **推理结果**

在单机单卡上使用MLU370-X4对训练了16000个step的checkpoint进行推理，其精度表现如下表所示。

Models |  Jit/Eager   | Supported Data Precision  | Batch Size | tumor_core  | peritumoral_edema | enhancing_tumor | MeanDice | WholeTumor 
------ | ------------ | ------------------------- | ---------- | ----------- | ----------------- | --------------- | -------- | ---------- 
UNet3D |     Eager    | FP32                      |     2      |  0.6475406289100647  |  0.7710946798324585 | 0.7304178476333618 | 0.716351052125295 | 0.8871155977249146





# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO



