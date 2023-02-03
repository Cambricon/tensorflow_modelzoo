**DLRM (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的DLRM网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
* [3.默认参数说明](#3-默认参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.结果展示](#5-结果展示) 
* [6.免责声明](#6-免责声明)
* [7.Release_Notes](#7-Release_Notes)


# 1. **模型概述**
DLRM网络是是Facebook在2019年提出的用于处理CTR问题的算法模型，原始论文为[Deep Learning Recommendation Model for
Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf)。

DLRM网络结构的代码实现可参考：[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/DLRM)。

# 2. **模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
DLRM | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


# 3. **默认参数说明**

| 参数 | 作用 | 默认值 |
|------|------|------|
| mode | 选择`train`来训练模型，选择`inference`来进行基准推理，选择`eval`来运行验证 | train |
| batch_size | 更改网络训练的batch_size | 65536 |
| valid_batch_size | 更改网络验证的batch_size，必须与训练batch_size相等 | 65536 |
| max_steps | 更改最终训练的step数目 | -1 |
| evals_per_epoch | 每个epoch的评估次数 | 1 |
| epochs | 更改最终训练的epochs数 | 1 |
| restore_checkpoint_path | 更改checkpoint目录或加载的checkpoint对象 |  |
| use_mlus | 是否使用MLU进行训练 | True |
| use_gpus | 是否使用GPU进行训练 | False |
| use_horovod | 控制是否使用horovod进行单机多卡训练 | True |
| print_freq | 调试打印之间的step数, 默认训练1000step显示一次log | 1000 |
| use_profiler | 是否使用profiler性能分析工具 | True |
| profiler_start_step | 使用性能分析工具时，从指定的step数开始保存数据 | 1 |
| profiler_steps | 使用性能分析工具时，指定保存数据的step数,默认保存1step的性能数据 | 1 |
| profiled_rank | 使用性能分析工具时，指定保存数据的卡号（默认为0卡，该参数主要用于性能测试） | 0 |

  
# 4. **快速使用**
下面将详细展示如何在 Cambricon TensorFlow2上完成DLRM的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪MLU300系列计算板卡，如需进行训练，则需装配MLU370-X8，若只需推理，则装配MLU370-S4/X4/X8均可；
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
# Modify the  YOUR_DOCKER_IMAGE_NAME according to your own environment.
# For instance, IMAGE_NAME=tensorflow2-1.12.1-x86_64-ubuntu18.04

IMAGE_NAME=YOUR_DOCKER_IMAGE_NAME
IMAGE_TAG=latest

export MY_CONTAINER="dlrm_tensorflow_modelzoo"

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

使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/Recommendation/DLRM` 目录。

**d)安装模型依赖项**

```bash
#安装依赖库, 需要有root权限进行apt install
pip install -r requirements.txt
# 安装性能测试工具，若不设置use_performance=True，则无需安装
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
cd tensorflow_modelzoo/tensorflow2/built-in/Recommendation/DLRM

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=your_docker_image_name
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便根据您的命名生成了一个名为`your_docker_image_name`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为your_docker_image_name
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
该DLRM脚本基于Criteo数据集训练，数据集下载：[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
需要将数据集进行预处理：
```bash
cd models/preproc
./prepare_dataset.sh CPU 15
```
**注意**：需将`env.sh`内的`DATA_DIR`的值改为`YOUR_DATA_PATH`。


## 4.3 **运行Run脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
DLRM  | TensorFlow2  | MLU370-X8  | FP32  | 8  | bash Horovod_DLRM_Float32_1E_8MLUs.sh
DLRM  | TensorFlow2  | MLU370-X8  | AMP  | 8  | bash Horovod_DLRM_AMP_1E_8MLUs.sh

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`batch_size`，`max_steps`，`use_amp`等，进入`run_scripts`目录后，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_DLRM_Float32_1E_8MLUs.sh
```

# 5. **结果展示**

**Training accuracy results: MLU370-X8**

DLRM推荐网络任务的训练精度通常用`accuracy`表征，在本仓库中，最终的训练精度由`AUC`表征。最终的训练精度如下所示： 

Models  | MLUs |  Mixed Precision Top1   | FP32 Top1 
----- | ----- | ----- | ----- |
DLRM | 8  | 0.802663 | 0.802576

# 6. **免责声明**
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7. **Release_Notes**
@TODO

