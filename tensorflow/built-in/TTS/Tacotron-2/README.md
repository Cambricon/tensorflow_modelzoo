**Tacotron2 (TensorFlow)**

本仓库是在MLU上基于TensorFlow1框架实现的Tacotron2网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-模型支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.默认参数说明](#3-默认参数说明)
  * [3.1模型训练参数说明](#31-模型训练参数说明)
  * [3.2模型推理参数说明](#32-模型推理参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.免责声明](#6-免责声明)
* [6.Release_Notes](#7-Release_Notes)


# 1. 模型概述
Tacotron2模型是端到端的TTS深度神经网络模型，原始论文为[Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)。

Tacotron2网络结构的代码实现可参考：[这里](https://github.com/Rayhane-mamah/Tacotron-2)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Tacotron2 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Jit/Eager Support 
----- | ----- | ----- | ----- | ----- | 
Tacotron2 | TensorFlow1  | MLU370X4/X8/S4  | FP32  |  Eager

# 3. 默认参数说明

## 3.1 **模型训练关键参数说明**

Tacotron2网络的训练参数在train.py中均设置了默认值，可通过run_scripts中的脚本传入相关参数值。
常用参数及含义如下表所示，更多参数可参考train.py。

| 参数     | 作用     | 默认值   |
|----------|----------|----------|
| tacotron_num_devices | 训练时的板卡数  | 1 |
| tacotron_batch_size | 训练的batch_size | 32  |
| tacotron_synthesis_batch_size  | 推理时的batchsize | 1  |
| device_type | 设备类型 | mlu |
| checkpoint_interval | ckpt保存的间隔 | 5000 |
| tacotron_train_steps  | 训练的代数  | 150000 |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
| input_dir | 输入数据集的文件夹 | training_data |
| tacotron_input | 输入数据集的列表文件 | training_data/train.txt |
| output_dir | 保存训练ckpt的目录 | output |
| use_horovod  | 训练是否使用horovod模式 | false |
| use_profiler | 为true则开启tensorboard | false |
| use_performance | 为true则开启性能测试模式 | false |


## 3.2 **模型推理参数说明**

Tacotron2网络的推理参数在synthesize.py中均设置了默认值，可通过run_scripts中的推理脚本传入相关参数值。
常用参数及含义如下表所示，更多参数可参考synthesize.py。

|   参数  |  作用  |  默认值  |
|---------|--------|----------|
| tacotron_num_devices | 训练时的板卡数  | 1 |
| tacotron_batch_size | 训练的batch_size | 32  |
| tacotron_synthesis_batch_size  | 推理时的batchsize | 1  |
| device_type | 设备类型 | mlu |
| taco_checkpoint | 指向保存checkpoint的路径 | None |
| mels_dir   | 包含mels的文件夹，用于使用wavenet合成音频  | tacotron_output/eval/ |

 
# 4. **快速使用**
下面将详细展示如何在 Cambricon TensorFlow1上完成Tacotron2的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-S4/X4/X8均可；
* Cambricon Driver >=v4.20.6；
* CNTensorFlow >= 1.15.5;
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

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=YOUR_IMAGE_TAG

export MY_CONTAINER="tf1_tacotron2_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/TTS/Tacotron-2` 目录。

**d)安装模型依赖项**

```bash
# 安装网络所需的依赖库
bash install_dependency.sh
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
cd tensorflow_modelzoo/tensorflow/built-in/TTS/Tacotron-2/  

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=tacotron2_network_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`tacotron2_network_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为tacotron2_network_image
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**

**(1)训练数据集准备**

该Tacotron2脚本基于LJSpeech-1.1训练，数据集下载：[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)。
下载数据集后，解压缩压缩文件，并将文件夹放入LJSpeech-1.1目录。

**(2)推理数据集准备**

该网络没有固定的推理数据集，可通过`--text_list`参数输入语句或者使用网络中默认的语句，可参考models/sentences.txt文件。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run 
----- | ----- | ----- | ----- | ----- | ----- | 
Tacotron2  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_Tacotron2_Float32_150000S_8MLUs.sh 
Tacotron2  | TensorFlow  | MLU370-X8  | AMP   | 8  | Horovod_Tacotron2_AMP_150000S_8MLUs.sh 
Tacotron2  | TensorFlow  | MLU370-X8  | FP32  | 1  | Tacotron2_Float32_150000S_1MLU.sh 

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`device_type`，`tacotron_batch_size`，`tacotron_train_steps`，`use_amp`等，按照如下命令即可开始from_scratch的Horovod分布式训练：
```bash
bash run_scripts/Horovod_Tacotron2_Float32_150000S_8MLUs.sh
```
训练过程中产生的模型文件及权重会保存至`output_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Tacotron2_Float32_150000S_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载tacotron2的tacotron_model.ckpt-18750进行finetune训练，训练1000step。
# 则tacotron_train_steps应设为19750，设置taco_checkpoint为tacotron_model.ckpt-18750模型所在的路径。

#!/bin/bash
dev_workspace=$(pwd)
if [[ $dev_workspace != *Tacotron* ]];
then
    echo "Please perform the training in the Tacotron2 workspace!"
    exit -1
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace="${dev_workspace}/.."
else
   workspace=$dev_workspace
fi
timestamp=$(date +%Y%m%d%H%M)
model_dir="${workspace}/Tacotron2_model_${timestamp}"

pushd "${workspace}"

source env.sh
bash install_dependency.sh

horovodrun -np 8 python train.py            \
           --tacotron_num_devices=1         \
           --tacotron_batch_size=32         \
           --tacotron_synthesis_batch_size=1\
           --taco_checkpoint=logs-Tacotron-2/taco_pretrained/     \
           --device_type=mlu                \
           --checkpoint_interval=1000       \
           --tacotron_train_steps=19750     \
           --use_amp=False                  \
           --input_dir=${DATA_DIR}          \
           --tacotron_input=${DATA_DIR}/train.txt    \
           --output_dir=mlu_model           \
           --use_horovod=True               \
           --use_profiler=False             \
           --use_performance=False
popd

```

**注意**：使用预训练模型进行finetune训练时，`tacotron_batch_size`，`tacotron_train_steps`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。

### 4.3.2 **一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种参数配置的脚本：`run_scripts/Infer_Tacotron2_Float32_1MLU.sh`，您可根据自己的需求修改该脚本内的`tacotron_synthesis_batch_size`，`taco_checkpoint`，完成修改后，按照如下命令运行即可分别以不同的参数与推理模式推理。

```bash
bash run_scripts/Infer_Tacotron2_Float32_1MLU.sh
```

目前支持的精度类型与推理模式组合以及运行环境如下所示：

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support| RUN |
|----- | ----- | ----- | ----- | ----- | ----- | 
Tacotron2   | TensorFlow  | MLU370-X4/X8/S4  | FP32   | Eager| bash run_scripts/Infer_Tacotron2_Float32_1MLU.sh |


# 5. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6. Release_Notes
@TODO
