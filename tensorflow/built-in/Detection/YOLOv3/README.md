**YOLOv3 (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的网络，支持训练与推理。

------------

**目录 (Table of Contents)**
- [1. 模型概述](#1-模型概述)
- [2. 模型支持情况](#2-模型支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
- [3. 默认参数配置](#3-默认参数配置)
- [4. 快速使用](#4-快速使用)
  - [4.1 依赖项检查](#41-依赖项检查)
  - [4.2 环境准备](#42-环境准备)
  - [4.3 运行Run脚本](#43-运行Run脚本)
- [5. 免责声明](#5-免责声明)
- [6. Release_Notes](#6-Release_Notes)


# 1. 模型概述


YOLOv3是目标检测网络，原始论文为[YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)。
YOLOv3网络结构的代码实现可参考：[这里](https://github.com/YunYang1994/tensorflow-yolov3)。

# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
YOLOv3 | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**
| Models   | Framework   | Supported MLU | Supported Data Precision | Jit/Eager Support |
|----------|-------------|---------------|--------------------------|-------------------|
| YOLOV3 | TensorFlow1 | MLU370-X4/X8  | FP32                     | Eager             |


# 3. 默认参数配置

| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 更改训练的batch_size | 8 |
| first_stage_epochs | 第一阶段的epoch数 | 40 |
| second_stage_epochs | 第二阶段的epoch数 | 60 |
| start_epoch | 开始的epoch数 | 1 |
| ckpt_path | 加载checkpoint路径 | checkpoint/yolov3_coco_demo.ckpt |
| use_profiler | 为True则开启性能模式 | False |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
| use_performance | 开启后本地生成summary文件夹，并在文件夹下的summary.json文件中记录性能 | False |
| finetune_step | 控制finetune训练的步数 | 0 |
| output_dir | checkpoint保存路径 | \ |
| hvd_device | 控制hvd运行的设备,可设置为mlu/gpu | mlu |
| use_horovod | 是否开启horovd | 单卡为False，多卡为True |

# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成YOLOv3的训练与推理。
## 4.1 **依赖项检查**
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

下载Cambricon TensorFlow1 docker镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow1_Image.tar.gz`

**b)启动容器**  

`run_docker.sh`示例如下，根据本地的镜像版本，修改如下示例中的`IMAGE_NAME`和`IMAGE_TAG`变量后再运行`bash run_docker.sh`即可启动容器。
```bash
#!/bin/bash

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Detection/YOLOv3` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../../../tensorflow2/built-in/tools/record_time/
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
cd tensorflow_modelzoo/tensorflow/built-in/Detection/YOLOv3

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=YOLOv3_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`YOLOv3_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为YOLOv3_image
# 2. 运行run_docker.sh
bash run_docker.sh
```

### 4.2.2 **数据集准备**
此demo基于COCO17数据集训练，数据集下载官网：[https://cocodataset.org/](https://cocodataset.org/)
具体可以点击[train2017](http://images.cocodataset.org/zips/train2017.zip)和[val2017](http://images.cocodataset.org/zips/val2017.zip)下载训练集和验证集，解压后可使用。

本地数据集目录结构请与下方保持一致：
``` bash
├── train2017
└── val2017
```

完成数据集准备后，根据数据集实际路径修改`env.sh`内的值。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

```bash
# 下载COCO weights
cd run_scripts
bash download_weights.sh
```

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
YOLOv3  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash YOLOv3_Float32_100E_1MLU.sh
YOLOv3  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_YOLOv3_Float32_100E_8MLUs.sh
YOLOv3  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_YOLOv3_AMP_100E_8MLUs.sh

在执行训练脚本之前需要先使用`download_weights.sh`脚本下载权重，所有脚本存放于`run_scripts`内，根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`output_dir`，`batch_size`等，按照上述命令即可开始from_scratch的分布式训练：

训练完成后，程序会输出训练精度`Average Precision`，并将训练过程中产生的模型文件及权重保存至`output_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_YOLOv3_AMP_100E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载checkpoint文件进行finetune，训练1000 step 则finetune_step应设为1000。
# 以加载checkpoint文件夹下yolov3_train_loss=27.8314.ckpt-4进行1000代的finetune为例。

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/yolov3_model_${timestamp}"

pushd "${work_dir}"

source env.sh
ANNOT_PATH="${work_dir}/models/data/dataset/coco17_train.txt"
sed -i s:^0:${TRAIN_FILE_LIST_PATH}0:g ${ANNOT_PATH}

horovodrun -np 8 python3 train.py \
    --batch_size=8 \
    --first_stage_epochs=40 \
    --second_stage_epochs=60 \
    --start_epoch=1 \
    --ckpt_path="checkpoint/yolov3_train_loss=27.8314.ckpt-4" \
    --finetune_step=1000 \
    --output_dir=$model_dir \
    --use_horovod=True \
    --hvd_device=mlu \
    --use_amp=True \
    --use_performance=False \
    --use_profiler=False
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **一键执行推理脚本**
本仓库提供了推理YOLOv3网络的脚本：`run_scripts/infer_run_eager_YOLOv3.sh`，需要用户手动传入`weight_file`参数，ckpt需要给到具体的file而不能只到dir，其中`weight_file`默认以`tensorflow_modelzoo/tensorflow/built-in/Detection/YOLOv3`为当前目录。具体参见`3. 默认参数配置`，具体示例如下：
```bash
cd run_scripts
修改脚本内weight_file参数为自己的ckpt file（例如yolov3_model/yolov3_train_loss=19.2270.ckpt-100）
bash infer_run_eager_YOLOv3.sh
```


# 5. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6. Release_Notes
@TODO



