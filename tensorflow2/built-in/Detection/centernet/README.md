**Centernet (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的Centernet网络，支持训练。


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
Centernet是一个基于Anchor-free的目标检测算法，输入是图片，输出是带有目标检测框的图片。原始论文为[Object as Points](https://arxiv.org/pdf/1904.07850.pdf)

Centernet网络的TensorFlow原生代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/official/projects/centernet)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Centernet | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

# 3. 默认参数配置

## 3.1 **模型训练参数说明**


Centernet模型的训练参数存在于`model_main_tf2.py`内，同时受到`mlu_centernet_config.config`及scripts/内的shell脚本的共同影响。

（1）scripts/内的shell脚本涉及到的常用参数及含义如下表所示：

<details>
<summary>展开查看</summary>


| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 训练的batch_size | 16   |
| num_train_steps | 不为0时，表示用户自主设定的每个epoch的步数。 | 0 |
| fine_tune_ckpt | 加载的checkpoint对象 | None|
| use_amp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| use_gpu| 是否使用gpu进行训练 | False |
| use_profiler| 是否支持tensorboard，若为True则表示| False |
| use_performance | 是否开启性能测试，若为True则表示开启，训练结束后可在summary/summary.json内读出throughput与e2e| False |

</details>

```

# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成Centernet的训练。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNTensorFlow >= 2.5.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 4.2 **环境准备**
### 4.2.1 **导入镜像**
下载Cambricon TensorFlow2 docker镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow2_Image.tar.gz`
### 4.2.2 **启动容器**
`run_docker.sh`示例如下，修改如下示例中的`YOUR_XXX`变量后再运行`bash run_docker.sh`即可启动容器。分布式训练时需使用多卡，单机多卡容器启动脚本示例如下：
<details>
<summary>展开查看</summary>
<pre><code>
#!/bin/bash
#below is a sample of run_docker.sh,
#modify the UPPERCASE var according to your own environment.

IMAGE_NAME=tensorflow2-x.y.z-x86_64-ubuntu18.04 #replace x.y.z with the real tf version number
IMG_TAG=latest
YOUR_PATH=$PWD
YOUR_PATH_INSIDE="/home/user_dir_inside"
export MY_CONTAINER="YOUR_DOCKER_NAME_8mlu"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER

if [ 0 -eq $num ];then
    xhost +
    docker run -it --name="${MY_CONTAINER}" \
     --net=host \
     --cap-add=sys_ptrace \
     --privileged=true \
     --shm-size="64g" \
     -v /usr/bin/cnmon:/usr/bin/cnmon \
     -v $PWD:/mnt \
     --device=/dev/cambricon_dev0 \
     --device=/dev/cambricon_dev1 \
     --device=/dev/cambricon_dev2 \
     --device=/dev/cambricon_dev3 \
     --device=/dev/cambricon_dev4 \
     --device=/dev/cambricon_dev5 \
     --device=/dev/cambricon_dev6 \
     --device=/dev/cambricon_dev7 \
     --device=/dev/cambricon_dev8 \
     --device=/dev/cambricon_dev9 \
     --device=/dev/cambricon_dev10 \
     --device=/dev/cambricon_dev11 \
     --device=/dev/cambricon_dev12 \
     --device=/dev/cambricon_dev13 \
     --device=/dev/cambricon_dev14 \
     --device=/dev/cambricon_dev15 \
     --device=/dev/cambricon_ctl \
     $IMAGE_NAME:$IMG_TAG  \
     /bin/bash
else
    docker start $MY_CONTAINER
    docker exec -ti --env COLUMNS=`tput cols` --env LINES=`tput lines` $MY_CONTAINER /bin/bash

fi
</code></pre>
</details>



### 4.2.3 **下载项目代码**

使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Detection/centernet` 目录。

### 4.2.4 **安装依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
pip install PATHTO/tensorflow_modelzoo/tensorflow2/built-in/tools/record_time
```

### 4.2.5 **数据集准备**
本仓库使用的是`COCO 2017`数据集。数据集下载：[https://cocodataset.org](https://cocodataset.org)
需要将数据集转换为tfrecord格式，可参见：[https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py)
在`scripts/`下运行`prepare_dataset.sh`(需要设置`DATASET`及`DATASET_LABEL`环境变量的值为本地对应数据集的路径)即可。
本地数据集目录结构需要与下方保持一致:
```bash
./data
└── coco2017_tfrecords
    ├── mscoco_label_map.pbtxt
    └── tfrecords
         ├── coco_train.record-00000-of-00100
         ├── coco_train.record-00001-of-00100
         ├── ...
         ├── coco_val.record-00008-of-00010
         └── coco_val.record-00009-of-00010
```

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
`scripts/`目录下提供了from_scratch的训练脚本。


Models | Framework | MLU | Data Precision | Cards | Run
----- | ----- | ----- | ----- | ----- | ----- |
Centernet | TensorFlow2 | MLU370-X8 | FP32 | 8 |Horovod_Centernet_Float32_8MLUs.sh
Centernet | TensorFlow2 | MLU370-X8 | AMP  | 8 |Horovod_Centernet_AMP_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`batch_size`，`steps`，`use_amp`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_Centernet_Float32_8MLUs.sh
```
# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

Centernet的训练精度可由验证集loss表征。

Models   | MLUs |Batch Size  | Steps  |Precision(FP32)  | Precision(Mixed Precision)  |
----- | ----- | ----- | ----- | ----- | ----- |
Centernet  | 8 | 8 | 140000 | 29.8 | 24.8


**Training performance results: MLU370-X8**

在运行`conformer_train.py`时候传入`--use_performance=True` 参数。
以下性能结果基于cambricon-tensorflow2(v1.12.1)取得。由于Centernet中能以fp16精度运行的算子较少，大量的算子仍以fp32精度运行，因此，数据类型转换（fp32转fp16）导致的耗时增加与算子与算子以fp16精度运行导致的耗时减少基本持平，从性能表现来看，便会出现混合精度训练的fps只比fp32精度训练的fps略高的情况。

Models   | MLUs |Batch Size  | Throughput(FP32)  | Throughput(Mixed Precision)  |  FP32 Training Time(100E) | Mixed Precision Training Time(100E)
----- | ----- | ----- | ----- | ----- | -----| -----|
Centernet  | 8  | 10 | 97.71 | 160.08 | N/A| N/A
Centernet  | 16 | 10 | 183.5 | 290.12 | N/A| N/A

# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO


