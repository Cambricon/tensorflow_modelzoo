**RetinaNet (TensorFlow1)**

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


RetinaNet是目标检测网络，原始论文为[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)。
RetinaNet网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/blob/v2.2.0/official/vision/detection/modeling/retinanet_model.py)。

# 2. 模型支持情况

## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
RetinaNet | TensorFlow1  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Jit/Eager Support 
----- | ----- | ----- | ----- | ----- | 
RetinaNet | TensorFlow1  | MLU370-X4/X8  | FP32  |  Eager


# 3. 默认参数配置

| 参数 | 作用 | 默认值 |
|------|------|------|
| strategy_type | 控制是否开启原生分布式，原生分布式不能与Horovod分布式同时开启，可选项为off、one_device、mirrored | off |
| num_mlus，num_gpus | 联合控制网络运行的设备，在mlu设备上运行需设置num_mlus=1,num_gpus=0；在gpu设备上运行需设置num_mlus=0,num_gpus=1 | num_mlus=1,num_gpus=0 |
| batch_size | 更改训练的batch_size | 8 |
| init_learning_rate | 初始学习率 | 单卡0.01，8卡0.08 |
| learning_rate_levels | 学习率更新范围 | 单卡[0.001,0.0001]，8卡[0.008,0.0008] |
| learning_rate_steps | 学习率步数间隔 | 单卡[439920,586560]，8卡[54990,73320] |
| iterations_per_loop | 一次estimator loop的迭代步数 | 单卡14664，8卡1833 |
| use_profiler | 为True则开启性能模式 | False |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
| use_performance | 开启后本地生成summary文件夹，并在文件夹下的summary.json文件中记录性能 | False |
| total_steps | 控制训练的步数 | 单卡733200，8卡91650 |
| model_dir | checkpoint保存路径 | \ |
| use_horovod | 是否开启horovd | 单卡为False，多卡为True |
| training_file_pattern | 训练数据集 | \ |
| eval_file_pattern | 推理数据集 | \ |
| val_json_file | 推理数据集信息 | \ |


# 4. 快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成retinanet网络的训练与推理。
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

export MY_CONTAINER="tf1_retinanet_tensorflow_modelzoo"

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/Detection/retinanet` 目录。

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
cd tensorflow_modelzoo/tensorflow/built-in/Detection/retinanet

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=retinanet_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`retinanet_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为retinanet_image
# 2. 运行run_docker.sh
bash run_docker.sh

```

### 4.2.2 **数据集准备**
此demo基于COCO17数据集训练，数据集下载：[https://cocodataset.org/](https://cocodataset.org/)
需要将数据集转换为tfrecord格式，可参见：[https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py)
本地数据集目录结构请与下方保持一致：
``` bash
├── coco_train.record-00000-of-00100
├── coco_train.record-00001-of-00100
├── ...
├── coco_val.record-00000-of-00010
├── coco_val.record-00001-of-00010
├── ...
└── instances_val2017.json
```

完成数据集准备后，根据数据集实际路径修改`env.sh`内的值。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
RetinaNet  | TensorFlow1  | MLU370-X8  | FP32  | 1  | bash Retinanet_Float32_1E_1MLU.sh
RetinaNet  | TensorFlow1  | MLU370-X8  | FP32  | 8  | bash Horovod_Retinanet_Float32_1E_8MLUs.sh
RetinaNet  | TensorFlow1  | MLU370-X8  | AMP  | 8  | bash Horovod_Retinanet_AMP_1E_8MLUs.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`model_dir`，`batch_size`等，按照上述命令即可开始from_scratch的分布式训练：

训练完成后，程序会输出训练精度`Average Precision`，并将训练过程中产生的模型文件及权重保存至`model_dir`指定的目录内。

若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Retinanet_Float32_1E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，加载checkpoint文件进行finetune，训练1000 step 则total_steps应设为当前step + 1000。
# 以加载model.ckpt-70000进行1000代的finetune为例， total_steps应设为 71000。

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/retinanet_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 train.py \
    --strategy_type=off \
    --num_mlus=1 \
    --num_gpus=0 \
    --mode=train \
    --batch_size=8 \
    --init_learning_rate=0.08 \
    --learning_rate_levels=[0.008,0.0008] \
    --learning_rate_steps=[54990,73320] \
    --iterations_per_loop=1833 \
    --use_amp=False \
    --total_steps=91650 \
    --training_file_pattern=$TRAIN_DATA_DIR \
    --eval_file_pattern=$EVAL_DATA_DIR \
    --val_json_file=$VAL_JSON_FILE \
    --model_dir=$model_dir \
    --use_horovod=True \
    --use_performance=False \
    --use_profiler=False
popd
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。


### 4.3.2 **一键执行推理脚本**
本仓库提供了推理retinanet网络的脚本：`run_scripts/infer_run_eager_Retinanet.sh`，需要用户手动修改`model_dir`参数，其中`model_dir`默认以`tensorflow_modelzoo/tensorflow/built-in/Detection/retinanet`为当前目录。具体参见`3. 默认参数配置`，具体示例如下：
```bash
cd run_scripts
bash infer_run_eager_Retinanet.sh
```


# 5. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6. Release_Notes
@TODO



