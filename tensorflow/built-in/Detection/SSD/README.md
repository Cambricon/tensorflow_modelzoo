**SSD (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的网络，支持训练与推理。

------------

**目录 (Table of Contents)**
- [1. 模型概述](#1-模型概述)
- [2. 模型支持情况](#2-模型支持情况)
- [3. 默认参数配置](#3-默认参数配置)
- [4. 快速使用](#4-快速使用)
  - [4.1 依赖项检查](#41-依赖项检查)
  - [4.2 环境准备](#42-环境准备)
  - [4.3 运行Run脚本](#43-运行Run脚本)
- [5. 结果展示](#5-结果展示)
- [6. 免责声明](#6-免责声明)
- [7. Release_Notes](#7-Release_Notes)


# 1. 模型概述


SSD是目标检测网络，原始论文为[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)。
SSD网络结构的代码实现可参考：[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD)。

# 2. 模型支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
SSD | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


# 3. 默认参数配置

| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 更改训练的batch_size | 32 |
| pipeline_config_path | 存放不同设备数的参数config | models/configs/ssd320_full_1mlus.config |
| num_steps | 训练步数 | 100000 |
| fine_tune_checkpoint | 加载finetune的checkpoint路径 | \ |
| fine_tune_checkpoint_type | 加载finetune的checkpoint的类型，可设置detection\classification | classification |
| model_dir | 存储checkpoint路径 | \ |
| use_profiler | 为True则开启性能模式 | False |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
| use_performance | 开启后本地生成summary文件夹，并在文件夹下的summary.json文件中记录性能 | False |
| hvd_device | 控制hvd运行的设备,可设置为mlu/gpu | mlu |
| use_horovod | 是否开启horovd | 单卡为False，多卡为True |

# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成SSD的训练与推理。
## 4.1 **依赖项检查**
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Detection/SSD` 目录。

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

# 3. 进入该网络目录
cd tensorflow_modelzoo/tensorflow/built-in/Detection/SSD

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=SSD_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`SSD_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为SSD_image
# 2. 运行run_docker.sh
bash run_docker.sh
```

### 4.2.2 **数据集准备**
此demo基于COCO17数据集训练，数据集下载：[https://cocodataset.org/#download](https://cocodataset.org/#download)
需要将数据集转换为tfrecord格式，可参见：[https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/dataset_tools/create_coco_tf_record.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/dataset_tools/create_coco_tf_record.py)
本地数据集目录结构请与下方保持一致：
``` bash
├── coco_train.record-00000-of-00100
├── coco_train.record-00001-of-00100
├── ...
├── coco_val.record-00000-of-00010
├── coco_val.record-00001-of-00010
├── ...
└── mscoco_label_map.pbtxt
```

完成数据集准备后，根据数据集实际路径修改`env.sh`内的值。


## 4.3 **运行Run脚本**

### 4.3.1 **执行训练脚本**

```bash
# 下载resnet backbone
cd run_scripts/
bash download_backbone.sh
```

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
SSD  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash SSD_Float32_1E_1MLU.sh
SSD  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_SSD_Float32_1E_8MLUs.sh
SSD  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_SSD_AMP_1E_8MLUs.sh


以上脚本均在run_scripts目录下。根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`model_dir`，`batch_size`等，按照上述命令即可开始from_scratch的分布式训练：

训练完成后，程序会输出训练精度`Average Precision`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_SSD_AMP_1E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载checkpoint文件进行finetune，训练1000 step 则num_steps应设为1000。
# 加载全网权重进行finetune训练前，需将checkpoint拷贝到./checkpoints文件夹中
# 从分类检查点恢复fine_tune_checkpoint_type应设为classification；从检测检查点恢复fine_tune_checkpoint_type应设为detection
# 以加载checkpoint文件夹下model.ckpt-100000进行1000代的finetune为例。

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/ssd_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 model_main.py \
    --batch_size=32 \
    --pipeline_config_path="models/configs/ssd320_full_8mlus.config" \
    --num_steps=1000 \
    --do_train=True \
    --fine_tune_checkpoint="./checkpoints/model.ckpt-100000" \
    --fine_tune_checkpoint_type="detection" \
    --model_dir=$model_dir \
    --use_horovod=True \
    --hvd_device=mlu \
    --use_amp=True \
    --use_performance=False \
    --use_profiler=False
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **一键执行推理脚本**
本仓库提供了推理SSD网络的脚本：`run_scripts/infer_run_eager_SSD.sh`，需要用户手动传入`model_dir`参数，其中`model_dir`默认以`tensorflow_modelzoo/tensorflow/built-in/Detection/SSD`为当前目录。具体参见`3. 默认参数配置`，具体示例如下：
```bash
cd run_scripts
# 修改脚本内model_dir参数为自己的ckpt路径
bash infer_run_eager_SSD.sh
```

# 5. 结果展示

**Training accuracy results: MLU370-X8**

图像检测任务的训练精度通常用`Average Precision`表征，在本仓库中，最终的训练精度由`Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]`表征。最终的训练精度如下所示：

Models  | MLUs |  Mixed Precision | FP32
----- | ----- | ----- | ----- |
SSD | 8  | 0.280 | 0.284

# 6. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. Release_Notes
@TODO



