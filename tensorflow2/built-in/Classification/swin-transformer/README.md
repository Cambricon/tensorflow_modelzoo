**Swin-Transformer (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的Swin-Transformer网络，支持训练与推理。

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
* [5.结果展示](#5-结果展示)  
  * [5.1推理结果展示](#51-推理结果展示)
* [6.免责声明](#6-免责声明) 
* [7.Release_Notes](#7-Release_Notes)


# 1. 模型概述
Swin-Transformer网络是Transformer的变种网络，原始论文为[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)。

Swin-Transformer网络结构的代码实现可参考：[这里](https://github.com/rishigami/Swin-Transformer-TF)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Swin-Transformer | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Jit/Eager Support 
----- | ----- | ----- | ----- | ----- | 
Swin-Transformer | TensorFlow2  | MLU370X4/X8/S4  | FP32  |  Eager

# 3. 默认参数说明

## 3.1 **模型训练关键参数说明**

swin-transformer网络的训练参数在swin-trainer.py中均设置了默认值，可通过run_scripts中的脚本传入相关参数值。
常用参数及含义如下表所示，更多参数可参考swin-trainer.py。

| 参数 | 作用 | 默认值 |
|------|------|------|
| dataset    | 使用的数据集   | None |
| batch_size | 更改训练的batch_size | None  |
| mode    | 网络运行的模式：包含训练和推理 | train_and_eval  |
| model_dir | 指向保存checkpoint的路径 | ./mlu_model |
| data_dir | 指向数据集的路径 | None |
| model_name  | 预加载的权重  | swin_large_224 |
| finetune_steps  | 微调的代数  | None |
| epochs  | 更改训练的epoch数目 | None |
| one_hot  | 是否只输出top1的精度值 | False |
| skip_eval  | 是否跳过推理部分 | False |
| validation_steps  | 推理步数设置 | None |
| use_horovod  | 训练是否使用horovod模式 | false |
| use_profiler | 为true则开启tensorboard | false |
| use_performance | 为true则开启性能测试模式 | false |
| use_dummy_synthetic_data | 是否使用合成数据集 | false |
| num_mlus | 使用mlu卡的数量 | 1 |
| num_gpus | 使用gpu卡的数量 | 1 |
| use_amp | 控制是否使用amp进行混合精度训练 | false |
| distribution_strategy | 基础分布式模式 | None |


## 3.2 **模型推理参数说明**

swin-transformer网络的训练参数在swin_infer.py中均设置了默认值，可通过run_scripts中的推理脚本传入相关参数值。
常用参数及含义如下表所示，更多参数可参考swin_infer.py。

| 参数 | 作用 | 默认值 |
|------|------|------|
| dataset    | 使用的数据集   | None |
| batch_size | 更改训练的batch_size | None  |
| mode    | 网络运行的模式：包含训练和推理 | train_and_eval  |
| checkpoint_file | 指向保存checkpoint的路径 | None |
| data_dir | 指向数据集的路径 | None |
| model_name  | 预加载的权重  | swin_large_224 |

 
# 4. **快速使用**
下面将详细展示如何在 Cambricon TensorFlow2上完成Swin-Transformer的训练与推理。
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
# Modify the  YOUR_IMAGE_NAME according to your own environment.
# For instance, IMAGE_NAME=tensorflow2-1.12.1-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_DOCKER_IMAGE_NAME
IMAGE_TAG=YOUR_DOCKER_IMAGE_TAG

export MY_CONTAINER="swin_tansformer_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow2/built-in/Classification/swin-transformer` 目录。

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

git clone https://gitee.com/cambricon/tensorflow_modelzoo.git

# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow2/built-in/Classification/swin-transformer 

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=swin-transformer_network_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`swin-transformer_network_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为swin-transformer_network_image
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**

**(1)训练数据集准备**
该Swin-Transformer脚本基于ImageNet2012训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
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
**(2)推理数据集准备**
本仓库使用ImageNet2012的val数据集作为推理数据集，完成数据集下载后，本地数据集目录结构应与下方一致：
```bash
val/
├── n01751748
│   ├── ILSVRC2012_val_00047384.JPEG
│   ├── ILSVRC2012_val_00048285.JPEG
│   ├── ......
├── n01443537
├── ......
└── val.txt
```
其中val.txt内容如下所示：
```bash
ILSVRC2012_val_00000001.JPEG n01751748
ILSVRC2012_val_00000002.JPEG n09193705
ILSVRC2012_val_00000003.JPEG n02105855
ILSVRC2012_val_00000004.JPEG n04263257
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Swin-Transformer  | TensorFlow2  | MLU370-X8  | FP32  | 8  | Horovod_Swin-Transformer_Float32_12E_8MLUs.sh
Swin-Transformer  | TensorFlow2  | MLU370-X8  | AMP | 8  | Horovod_Swin-Transformer_AMP_12E_8MLUs.sh 
Swin-Transformer  | TensorFlow2  | MLU370-X8  | FP32 | 1  | Swin-Transformer_Float32_12E_1MLU.sh 

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`epochs`，`np`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
Horovod_Swin-Transformer_Float32_90E_8MLUs.sh
```
训练完成后，程序会输出训练精度`accuracy`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Swin-Transformer_Float32_12E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载swin-transformer的model.ckpt-0010进行finetune训练，训练1000 step
# 则finetune_steps应设为1000，设置finetune_checkpoint为加载的checkpoint节点。

#!/bin/bash
dev_workspace=$(pwd)
if [[ $dev_workspace != *swin* ]];
then
    echo "Please perform the training in the swin-transformer workspace!"
    exit -1
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace="${dev_workspace}/.."
else
   workspace=$dev_workspace
fi
DATA_DIR=YOUR_DATA_PATH
MODEL_DIR=YOUR_MODEL_DIR
CKPT_PATH=YOUR_CHECKPOINT_PATH

pushd "${workspace}"

source env.sh

horovodrun -np 8 python swin_trainer.py   \
         --dataset=imagenet               \
         --mode=train_and_eval            \
         --data_dir=$DATA_DIR             \
         --model_dir=$MODEL_DIR           \
         --model_name=swin_large_224      \
         --batch_size=12                  \
         --finetune_steps=1000            \
         --finetune_checkpoint=$CKPT_PATH \
         --epochs=11                      \
         --one_hot=False                  \
         --skip_eval=True                 \
         --validation_steps=0             \
         --use_horovod=True               \
         --use_profiler=False             \
         --use_performance=False          \
         --use_dummy_synthetic_data=False \
         --num_mlus=1                     \
         --num_gpus=0                     \
         --use_amp=False                  \
         --distribution_strategy=mirrored

popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。

### 4.3.2 **一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种参数配置的脚本：`run_scripts/multi_infer_run.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`quant_precision`，完成修改后，按照如下命令运行即可分别以不同的参数与推理模式推理。

目前支持的精度类型与推理模式组合以及运行环境如下所示：

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support| RUN |
|----- | ----- | ----- | ----- | ----- | ----- | 
Swin-Transformer   | TensorFlow2  | MLU370-X4/X8/S4  | FP16/FP32   | Eager| bash infer_run_eager_fp32_bsz_12.sh |


# 5. **结果展示**

## 5.1 **推理结果展示**

**Infering  results: MLU370-X4**

在MLU370-X4单卡上进行推理，推理结果如下：

Models   | Jit/Eager   |  Data Precision|Batch Size  | top1| 
----- | ----- | ----- | ----- | -----  
Swin-Transformer | Eager |  FP32 | 12 | 0.8521|

# 6. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. Release_Notes
@TODO
