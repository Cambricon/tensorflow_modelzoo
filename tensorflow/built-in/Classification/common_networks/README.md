**ResNet50 (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的网络，共支持ResNet18、ResNet50、ResNet101、DenseNet201、Vgg16、Vgg19、InceptionV2、InceptionV3、AlexNet、MobilenetV2十种模型，支持训练与推理。

------------

**目录 (Table of Contents)**
- [1. 模型概述](#1-模型概述)
- [2. 模型支持情况](#2-模型支持情况)
- [3. 默认参数配置](#3-默认参数配置)
- [4.快速使用](#4-快速使用)
  - [4.1 依赖项检查](#41-依赖项检查)
  - [4.2 环境准备](#42-环境准备)
  - [4.3 运行Run脚本](#43-运行Run脚本)
- [5.免责声明](#5-免责声明)
- [6.Release_Notes](#6-Release_Notes)


# 1. 模型概述


ResNet18、ResNet50、ResNet101网络都是残差卷积网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。
ResNet网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/model/legacy/image_classification/resnet)。

DenseNet201网络是密集连接卷积网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1608.06993.pdf)。
DenseNet201网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)。

Vgg16与Vgg19网络的原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)。
Vgg16网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py)。
Vgg19网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py)。

InceptionV2与InceptionV3网络的原始论文为[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)。
InceptionV2网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/inception_resnet_v2.py)。
InceptionV3网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py)。

AlexNet网络的原始论文为[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)。
AlexNet网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/slim/nets/alexnet.py)。

MobilenetV2网络的原始论文为[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)。
MobilenetV2网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py)。

# 2. 模型支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet18 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
ResNet50 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
ResNet101 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
DenseNet201 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
Vgg16 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
Vgg19 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
InceptionV2 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
InceptionV3 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
AlexNet | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested
MobilenetV2 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested 


# 3. 默认参数配置

| 参数 | 作用 |
|------|------|
| batch_size | 更改训练的batch_size |
| model_type | 修改训练的网络名称，选项有resnet18、resnet50、resnet101、vgg16、vgg19、alexnet、inceptionv2、inceptionv3、mobilenetv2、densenet201 |
| model_dir | 指向保存checkpoint的路径 |
| data_dir | 指向数据集的路径 |
| epochs | 更改训练的epoch数目 |
| use_profiler | 为True则开启tensorboard |
| use_amp | 控制是否使用amp进行混合精度训练 |
| skip_eval | 是否跳过推理阶段 |
| use_performance | 开启后本地生成summary文件夹，并在文件夹下的summary.json文件中记录性能 |
| distribution_strategy | 控制是否开启原生分布式，原生分布式不能与Horovod分布式同时开启 |
| num_mlus，num_gpus | 联合控制网络运行的设备，在mlu设备上运行需设置num_mlus=1,num_gpus=0；在gpu设备上运行需设置num_mlus=0,num_gpus=1 |
| validation_steps | 控制推理的步数 |
| use_qat | 是否开启qat训练，目前仅支持resenet50、mobilenetv2两个网络 |
| inter_op_threads | 设置OP间并发线程数 |
| intra_op_threads | 设置OP内并发线程数 |
| finetune_checkpoint | 指向加载checkpoint训练的路径 |
| finetune_steps | 与finetune_checkpoint连用，控制网络finetune的步数 |  


# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成分类网络的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Classification/common_networks` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd tensorflow_modelzoo/tensorflow2/built-in/tools/record_time/
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
cd tensorflow_modelzoo/tensorflow/built-in/Classification/common_networks

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

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
ResNet18  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash ResNet18_Float32_100E_1MLU.sh
ResNet18  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_ResNet18_Float32_100E_8MLUs.sh
ResNet18  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_ResNet18_AMP_100E_8MLUs.sh
ResNet50  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash ResNet50_Float32_100E_1MLU.sh
ResNet50  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_ResNet50_Float32_100E_8MLUs.sh
ResNet50  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_ResNet50_AMP_100E_8MLUs.sh
ResNet101  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash ResNet101_Float32_100E_1MLU.sh
ResNet101  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_ResNet101_Float32_100E_8MLUs.sh
ResNet101  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_ResNet101_AMP_100E_8MLUs.sh
Vgg16  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash Vgg16_Float32_100E_1MLU.sh
Vgg16  | TensorFlow1 | MLU370-X8  | FP32  | 8  | bash Horovod_Vgg16_Float32_100E_8MLUs.sh
Vgg16  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_Vgg16_AMP_100E_8MLUs.sh
Vgg19  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash Vgg19_Float32_100E_1MLU.sh
Vgg19  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_Vgg19_Float32_100E_8MLUs.sh
Vgg19  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_Vgg19_AMP_100E_8MLUs.sh
DenseNet201  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash DenseNet201_Float32_140E_1MLU.sh
DenseNet201  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_DenseNet201_Float32_140E_8MLUs.sh
DenseNet201  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_DenseNet201_AMP_140E_8MLUs.sh
InceptionV2  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash InceptionV2_Float32_100E_1MLU.sh
InceptionV2  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_InceptionV2_Float32_100E_8MLUs.sh
InceptionV2  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_InceptionV2_AMP_100E_8MLUs.sh
InceptionV3  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash InceptionV3_Float32_100E_1MLU.sh
InceptionV3  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_InceptionV3_Float32_100E_8MLUs.sh
InceptionV3  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_InceptionV3_AMP_100E_8MLUs.sh
AlexNet  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash AlexNet_Float32_100E_1MLU.sh
AlexNet  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_AlexNet_Float32_100E_8MLUs.sh
AlexNet  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_AlexNet_AMP_100E_8MLUs.sh
MobilenetV2  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash MobilenetV2_Float32_120E_1MLU.sh
MobilenetV2  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_MobilenetV2_Float32_120E_8MLUs.sh
MobilenetV2  | TensorFlow1  | MLU370-X8  | AMP | 8  | bash Horovod_MobilenetV2_AMP_120E_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`data_dir`，`batch_size`，`epochs`，`np`等，按照上述命令即可开始from_scratch的分布式训练：

训练完成后，程序会输出训练精度`accuracy`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_ResNet18_Float32_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载已经训练了50个epoch的checkpoint文件进行finetune，训练1000 step
# 则finetune_steps应设为1000，epochs应设为51

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet18_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=resnet18 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=128 \
    --epochs=51 \
    --start_epoch=0 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=True \
    --skip_eval=False \
    --finetune_steps=1000 \
    --finetune_checkpoint="your_ckpt_dir" \
    --validation_steps=0 \
    --use_qat=False \
    --inter_op_threads=0 \
    --intra_op_threads=0 \
    --use_dummy_synthetic_data=False
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **一键执行推理脚本**
为了遍历多种网络与输入规模，本仓库还提供了一键执行多种参数配置的脚本：`run_scripts/infer_run_eager_networks.sh`，需要用户手动传入`model_type`和`model_dir`两个参数，其中`model_dir`默认以`tensorflow_modelzoo/tensorflow/built-in/Classification/common_networks`为当前目录。具体参见`3. 默认参数配置`，以resnet18为例，具体示例如下：
```bash
cd run_scripts
bash infer_run_eager_networks.sh --model_type resnet18 --model_dir=resnet18_model_dir
```


# 5. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6. Release_Notes
@TODO



