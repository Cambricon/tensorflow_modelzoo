**ResNet50 (TensorFlow)**

本仓库是在MLU上基于TensorFlow框架实现的网络，共支持ResNet50、ResNet18、ResNet101、DenseNet201、Vgg16、Vgg19六种模型，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.模型支持情况](#2.支持情况)
  * [2.1训练模型支持情况](#2.1训练模型支持情况)
  * [2.2推理模型支持情况](#2.2推理模型支持情况)
* [3.默认参数配置](#3.默认参数配置)
  * [3.1模型训练默认参数配置](#3.1模型训练默认参数配置)
  * [3.2模型推理默认参数配置](#3.2模型推理默认参数配置)  
* [4.快速使用](#4.快速使用)
  * [4.1依赖项检查](#4.1依赖项检查)
  * [4.2环境准备](#4.2环境准备)
  * [4.3运行Run脚本](#4.3运行Run脚本)
* [5.结果展示](#5.结果展示)
  * [5.1训练结果](#5.1训练结果)
  * [5.2推理结果](#5.2推理结果)
* [6.免责声明](#6.免责声明) 
* [7.Release notes](#7.Release_Notes)


# 1. 模型概述


ResNet18、ResNet50和ResNet101网络都是残差卷积神经网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。
ResNet网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/model/legacy/image_classification/resnet)。

DenseNet201网络是密集连接卷积网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1608.06993.pdf)。
DenseNet201网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)。

Vgg16与Vgg19网络的原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)。
Vgg16网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py)。
Vgg19网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet18 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
ResNet50 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
ResNet101 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
DenseNet201 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
Vgg16 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
Vgg19 | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision   | 


注意，此处`Jit`表示使用`TFMM`的方式进行推理，即是用`TensorFlow2-MagicMind`作为底层实现后端进行推理。

# 3. 默认参数配置

## 3.1 **模型训练默认参数配置**

| 参数 | 作用 |
|------|------|
| batch_size | 更改训练的batch_size |
| model_dir | 指向保存checkpoint的路径 |
| data_dir | 指向数据集的路径 |
| epochs | 更改训练的epoch数目 |
| use_profiler | 为True则开启tensorboard |
| use_amp | 控制是否使用amp进行混合精度训练 |
| skip_eval | 是否跳过推理阶段 |
| enable_tensorboard | 控制是否开启tensorboard，并记录性能 |
| distribution_strategy | 控制是否开启原生分布式，原生分布式不能与Horovod分布式同时开启 |
| num_mlus，num_gpus | 联合控制网络运行的设备，在mlu设备上运行需设置num_mlus=1,num_gpus=0；在gpu设备上运行需设置num_mlus=0,num_gpus=1 |
  
## 3.2 **+ 模型推理默认参数说明**
<span id= "jump1"></span>
### 3.2.1 **+ 模型推理常用参数说明**

推理的公共参数(如量化相关的参数)都在`../../tools/infer_flags.py`内，非推理独有的参数则在`model/resnet.py`内的common函数。程序运行时会解析并读取该脚本内的所有参数。
大部分参数提供了默认值，这些参数的详细含义将在稍后给出。
我们根据常用的参数组合，在`scripts/`下提供了若干个常用的脚本，如`infer_run_eager_fp32_bsz_4.sh`，`infer_run_jit_fp32_bsz_4.sh`，在使用这些脚本之前，您需要根据当前环境修改如下常用参数：
```bash
data_dir#推理数据集路径，例如/home/data/imagenet/val，该数据集的目录结构需按照下文中的[数据集准备]一节的要求布置。
run_eagerly#是否使用eager模式进行推理。0表示不使用，1表示使用。默认为0。非eager模式也称为jit模式，即使用TFMM（TensorFlow2-MagicMind）进行推理，eager模式即基于TensorFlow框架的推理。支持外界直接传参。
batch_size#推理时的batch大小，默认为1。支持外界直接传参。
quant_precision#推理精度类型，默认为fp32，可选值为[fp16,fp32]其中之一。支持外界直接传参。
enable_dim_range#默认为 1，该参数仅在jit模式下有效。TFMM支持同一份模型使用不同的输入形状进行推理，当模型输入在某些维度长度可变时（例如batch,height,width,channel中的batch可变），开启该选项后可使推理性能达到更优。关于该参数更具体的解释与使用请参阅Cambricon-TensorFlow-MagicMind用户手册。
```
完成上述参数修改后，再运行`bash infer_run_eager_fp32_bsz_4.sh`即可开始推理。
这些脚本也支持通过直接传参的方式修改`batch_size`，`quant_precision`，`run_eagerly`。
例如运行`bash  infer_run_eager_fp32_bsz_4.sh -b 16`即可将`batch_size`从4改为16，且保持其他参数不变。

### 3.2.2 **+ 模型推理其他参数说明**

除了上述常用参数之外，`infer_run_eager_fp32_bsz_4.sh` （infer_run_jit_fp32_bsz_4.sh与之类似）还有如下参数可供使用与修改：
```bash
native_savedmodel_dir#该参数表示原生TensorFlow Keras原始模型（savedmodel格式）的存放路径，需要传入绝对路径
converted_savedmodel_dir#该参数表示转换为TFMM模型（savemodel格式）的存放路径，需要传入绝对路径
imagenet_label_file#imagenet标签文件的绝对路径，通常是${data_dir}/val.txt
result_path#推理结果保存路径，需要为绝对路径
net_name#网络名称，本仓库内为resnet50
visible_devices#对于多卡用户，设置程序运行在哪张卡上，默认为 0
top_k#用于输出推理概率前top_k大的结果
warmup_count#正式推理之前的热身推理次数。初次推理通常需要进行一定的初始化工作，为使推理时统计的性能数据更精确，通常不会计入第一次推理耗时。默认为1。
-----------------------------------
#如下参数只在jit模式时被使用

calibration_data_dir#用于量化校准的数据的绝对路径，图片格式需为JPEG。
quantize_algo#量化算法，可选值为['linear','eqnm']其中之一，默认值为'linear'
quantize_type#量化类型，可选值为['symm','asym']其中之一，即对称量化或非对称量化，默认值为'symm'
quantize_granularity#量化维度， 输入为 ['per_tensor', 'per_axis'] 其中之一，即指定逐tensor量化还是逐维度量化，默认为 'per_tensor'
opt_config#TF2MM模型优化性能选项，目前支持的输入为 [conv_scale_fold,type64to32_conversion] 如果想需要设置多个，用逗号 ',' 隔开
}

```
若要在脚本中使用过更多的参数，则需在`scripts/*.sh`脚本中新增对应的变量，再参照例如`quant_precision`的方式传入`resnet_infer.py`.
  
  
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Classification/common_networks` 目录。

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
cd tensorflow_modelzoo/tensorflow2/built-in/Classification/common_networks

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=common_networks_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`common_networks_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为common_networks_image
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
ResNet18  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_ResNet18_Float32_100E_1MLU.sh
ResNet18  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_ResNet18_Float32_100E_8MLUs.sh
ResNet18  | TensorFlow  | MLU370-X8  | AMP  | 8  | Horovod_ResNet18_AMP_100E_8MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_ResNet50_Float32_90E_1MLU.sh
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_ResNet50_Float32_90E_8MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | AMP  | 8  | Horovod_ResNet50_AMP_90E_8MLUs.sh
ResNet101  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_ResNet101_Float32_100E_1MLU.sh
ResNet101  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_ResNet101_Float32_100E_8MLUs.sh
ResNet101  | TensorFlow  | MLU370-X8  | AMP | 8  | Horovod_ResNet101_AMP_100E_8MLUs.sh
Vgg16  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_Vgg16_Float32_100E_1MLU.sh
Vgg16  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_Vgg16_Float32_100E_8MLUs.sh
Vgg16  | TensorFlow  | MLU370-X8  | AMP  | 8  | Horovod_Vgg16_AMP_100E_8MLUs.sh
Vgg19  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_Vgg19_Float32_100E_1MLU.sh
Vgg19  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_Vgg19_Float32_100E_8MLUs.sh
Vgg19  | TensorFlow  | MLU370-X8  | AMP  | 8  | Horovod_Vgg19_AMP_100E_8MLUs.sh
DesNet201  | TensorFlow  | MLU370-X8  | FP32  | 1  | Horovod_DesNet201_Float32_140E_1MLU.sh
DesNet201  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_DesNet201_Float32_140E_8MLUs.sh
DesNet201  | TensorFlow  | MLU370-X8  | AMP | 8  | Horovod_DesNet201_AMP_140E_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`train_steps`，`np`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_ResNet18_Float32_8MLUs.sh
```
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
data_dir=YOUR_DATA_PATH/transformer_data
ckpt_file=YOUR_PATH/mlu_model

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 classifier_trainer.py \
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
    --finetune_steps=0 \
    --finetune_checkpoint=$ckpt_file \
    --enable_tensorboard=False \
    --datasets_num_private_threads=0
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **+ 一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种参数配置的脚本：`scripts/multi_infer_run.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`quant_precision`，完成修改后，按照如下命令运行即可分别以不同的参数与推理模式（eager/jit）推理。
```bash
bash multi_infer_run.sh

```

目前支持的精度类型与推理模式组合以及运行环境如下所示：

Models  | Framework  | MLU   | Data Precision  |  Jit/Eager |
----- | ----- | ----- | ----- | ----- | ----- |



# 5. **结果展示**

## 5.1 **训练结果**

** Training accuracy results: MLU370-X8**

Models  | MLUs | Total Batch Size  |  Mixed Precision Top1   | FP32 Top1 
----- | ----- | ----- | ----- | ----- |
ResNet18 | 8  | MLU370-X8  |   | 
ResNet50 | 8  | MLU370-X8  |   | 
ResNet101 | 8  | MLU370-X8  |   | 
DenseNet201 | 8  | MLU370-X8  |   | 
Vgg16 | 8  | MLU370-X8  |   | 
Vgg19 | 8  | MLU370-X8  |   | 

** Training performance results: MLU370-X8**

Models   | MLUs   |  Total Batch Size  | Throughput(FP32)  | Throughput(Mixed Precision) 
----- | ----- | ----- | ----- | ----- | -----
ResNet18 | 1  | 128  |   |  
ResNet50 | 1  |   |   |  
ResNet101 | 1  |   |   |   
DenseNet201 | 1  |   |   |  
Vgg16 | 1  |   |   | 
Vgg19 | 1  |   |   |  


## 5.2  **+ 推理结果**


###  Infering  results: MLU370-X4

Models | mode   | precision  | batch_size|top1/top5  | hardware_fps  
----- | ----- | ----- | ----- | ----- | ----- 



# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO



