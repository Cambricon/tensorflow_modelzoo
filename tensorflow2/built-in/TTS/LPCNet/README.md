**LPCNet (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架使用Keras API接口实现的LPCNet网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.模型支持情况](#2.支持情况)
  * [2.1训练模型支持情况](#2.1训练模型支持情况)
* [3.默认参数说明](#3.默认参数说明)
  * [3.1模型训练参数说明](#3.1模型训练参数说明)
* [4.快速使用](#4.快速使用)
  * [4.1依赖项检查](#4.1依赖项检查)
  * [4.2环境准备](#4.2环境准备)
  * [4.3运行Run脚本](#4.3运行Run脚本)
* [5.结果展示](#5.结果展示)
  * [5.1训练结果](#5.1训练结果)
* [6.免责声明](#6.免责声明) 
* [7.Release notes](#7.Release_Notes)


# 1. 模型概述
LPCNet是一个神经网络声码器，用于语音合成，属于TTS系统的一部分，输入是声学特征向量，输出结果为音频文件。原始论文为[LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846)。

LPCNet网络的原生代码实现可参考：[这里](https://github.com/xiph/LPCNet)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
LPCNet | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


# 3. 默认参数配置

## 3.1 **模型训练参数说明**

## 3.1.1 **模型训练常用参数说明**

<details>
<summary>展开查看</summary>


| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 训练的batch_size | 16   |
| model_dir | 保存checkpoint的路径 | ./mlu_model |
| output | 训练得到的模型文件，默认为h5格式 | ./mlu_model |
| data | pcm音频文件的路径，用于训练。若无，参考4.2.6小节的指引生成。| data.u8 |
| features | 声学特征文件的路径，用于训练。若无，参考4.2.6小节的指引生成。| features.f32 |
| epochs | 训练的epoch数目 | 120 |
| lr | 训练初始学习率 | 0.001 |
| enable_tensorboard | 为True则开启tensorboard | False |
| use_amp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| use_performance | 是否开启性能测试，若为True则表示开启，训练结束后可得到throughput与e2e| False |
  
</details>


  
# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成LPCNet的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNTensorFlow >= 2.5.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 环境准备
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

使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/TTS/LPCNet` 目录。

**d)安装模型依赖项**

```bash
#安装依赖库, 需要有root权限进行apt install
cd models
sudo apt-get update
sudo apt-get install autoconf automake libtool sox ffmpeg
# 安装性能测试工具，若不设置use_performance=True，则无需安装
cd ../../tools/record_time/
pip install .
```

 **e)编译源代码**
编译源代码以便构建模型，**以下步骤只需在首次运行前执行一次**。

```bash
#在LPCNet/models目录内构建模型文件，并编译
./autogen.sh
./configure
make
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
cd tensorflow_modelzoo/tensorflow2/built-in/TTS/LPCNet

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=lpcnet_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`lpcnet_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为lpcnet_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
正式训练之前，需要准备相应的训练数据集。原始训练材料可从[此处](http://www-mmsp.ece.mcgill.ca/Documents/Data/)获得。下载`16k-LP7.zip`并解压，随后使用本仓库内的`src/concat.sh`将`16k-LP7`内的wav文件拼接成`input.s16`：
```bash
cd 16k-LP7
bash YOUR_PATH/src/concat.sh
```
在上一步的源码编译后，在仓库根目录下将产生一个`dump_data`的二进制工具，使用`./dump_data -h`可查看具体用法。
```bash
usage: ./dump_data -train <speech> <features out> <pcm out>
  or   ./dump_data -test <speech> <features out>
```
为了产生LPCNet实际可用的训练文件`features.f32` 与`data.u8`，需使用如下命令：
```bash
./dump_data -train path/to/16k-LP7/input.s16 features.f32 data.u8
```

**注意**：假设生成的`features.f32` 与`data.u8`路径为`YOUR_DATA_PATH`，则此时还需将`env.sh`内的`DATA_DIR`的值改为`YOUR_DATA_PATH`。

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
`run_scripts/`目录下提供了from_scratch的训练脚本。


Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
LPCNet| TensorFlow2  | MLU370-X8  | Float32  | 8  |Horovod_LPCNet_Float32_120E_8MLUs.sh
LPCNet  | TensorFlow2  | MLU370-X8  | AMP  | 8  |Horovod_LPCNet_AMP_120E_8MLUs.sh
LPCNet  | TensorFlow2  | MLU370-X8  | Float32  | 1  |LPCNet_Float32_120E_1MLU.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`batch_size`，`epochs`，`use_amp`等，按照如下命令即可开始from_scratch训练：
```bash
bash Horovod_LPCNet_Float32_120E_8MLUs.sh
```
若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_LPCNet_Float32_120E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载第120个epoch的checkpoint文件进行finetune训练1000代（finetune 1个epoch，每个epoch迭代1000 step）：
horovodrun -np 8  python lpcnet_train.py  \ 
--num_mlus=1 --num_gpus=0 --batch_size=16  --epochs=1 \ 
--steps_per_epoch=1000 --start_epoch=121 \
--finetune="YOUR_PATH/LPCNet/mlu_model_384_120.h5"  \ 
--use_amp=False --output=mlu_model --model_dir=mlu_model \
--use_performance=False --use_horovod=True \
--features=YOUR_PATH/16k-LP7/features.f32 \
--data=YOUR_PATH/16k-LP7/data.u8 --enable_tensorboard=False 
```
**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`需与from_scratch得到该预训练模型的参数一致，否则无法正常训练。


# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

LPCNet属于生成模型，不像分类网络，检测网络等判别模型，LPCNet无表征精度的客观指标，通常使用直接听生成的音频后再给出主观评分（MOS评分）的方式来评价合成的语音的质量。

**Training performance results: MLU370-X8**

为了在训练结束后获得性能测试结果，还需：  

在运行`lpcnet_train.py`时候传入`--use_performance=True` 参数。
以下性能结果基于cambricon-tensorflow2(v1.12.1)取得。由于LPCNet中能以fp16精度运行的算子较小，大量的算子仍以fp32精度运行，因此，数据类型转换导致的耗时增加无法被fp16更快运行导致的耗时减小完全覆盖，从性能表现来看，便会出现混合精度训练的fps不如fp32精度训练的fps。

Models   | MLUs |Batch Size  | Throughput(FP32)  | Throughput(Mixed Precision)  
----- | ----- | ----- | ----- | ----- | 
LPCNet  | 1  |16| 3.00  | 2.76  
LPCNet  | 4  |64| 11.35  | 9.98 
LPCNet  | 8  |128| 19.06  | 17.03   




# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO


