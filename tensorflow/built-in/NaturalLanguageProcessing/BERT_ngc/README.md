**Bert (TensorFlow1)**

本仓库是在MLU上基于TensorFlow1框架实现的Bert-NGC网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.默认参数说明](#3-默认参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.免责声明](#5-免责声明)
* [6.Release_Notes](#6-Release_Notes)


# 1. 模型概述
Bert是基于Transformer的一个网络模型，由多个Transformer的Encoder堆叠而成。Bert的创新之处主要在于在预训练时，通过使用Masked LM 和 Next Sentence Prediction两种方法捕捉了词语与句子级别的表征信息。在实际使用中，已完成预训练的Bert模型再加上特定的下游网络，如CRF，即可完成NLP的下游任务，如机器翻译，文本分类等。
原始论文为[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 。

本仓库使用Bert的开源预训练模型进行question-answering下游任务，需先finetune。其原生代码实现可参考[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)。

# 2. 模型支持情况

## 2.1 **训练模型支持情况**
| Models   | Framework   | Supported MLU | Supported Data Precision | Multi-GPUs | Multi-Nodes |
|----------|-------------|---------------|--------------------------|------------|-------------|
| Bert-NGC | TensorFlow1 | MLU370-X8     | FP16/FP32                | Yes        | Not Tested  | 

## 2.2 **推理模型支持情况**
| Models   | Framework   | Supported MLU | Supported Data Precision | Jit/Eager Support |
|----------|-------------|---------------|--------------------------|-------------------|
| Bert-NGC | TensorFlow1 | MLU370-X4/X8  | FP32                     | Eager             |

# 3. 默认参数说明
| 参数                   | 作用                                           | 默认值                                |
|------------------------|------------------------------------------------|------------------------------------|
| hvd_device             | 用于Horovod训练的设备                          | mlu                                |
| train_steps            | 训练的步数                                     | 0                                  |
| bert_config_file       | 与预训练的BERT模型相对应的json配置文件         | "${MODEL_DIR}/bert_config.json"    |
| vocab_file             | 训练BERT模型的词汇文件                         | "${MODEL_DIR}/vocab.txt"           |
| output_dir             | ckpt保存路径                                   | bert_model_${timestamp}            |
| dllog_path             | dllogger写入的文件名                           | "./bert_dllog.json"                |
| train_file             | 用于训练的SQuAD json                           | "${SQUAD_DIR}/train-v1.1.json"     |
| predict_file           | 用于预测的SQuAD json                           | "${SQUAD_DIR}/SQuAD/dev-v1.1.json" |
| eval_script            | 用于计算f1和exact_match的SQuAD evaluate.py文件 | "${SQUAD_DIR}/evaluate-v1.1.py"    |
| init_checkpoint        | 初始ckpt(通常来自预先训练的BERT模型)           | "${MODEL_DIR}/model.ckpt-5474"     |
| do_lower_case          | 是否对输入的文本进行小写                       | False                              |
| max_seq_length         | WordPiece标记化后的最大输入序列总长度          | 384                                |
| doc_stride             | 长文件分割后彼此之间的跨度                     | 128                                |
| do_train               | 是否进行训练                                   | True                               |
| do_predict             | 是否进行预测                                   | False                              |
| train_batch_size       | 训练的批大小                                   | 8                                  | 
| use_horovod            | 是否使用Horovod分布式                          | False                              |
| use_profiler           | 是否使用tfporf工具                             | False                              |
| use_performance        | 是否进行性能测试                               | False                              |
| inter_op_threads       | OP间并发线程数                                 | 0                                  |
| intra_op_threads       | OP内并发线程数                                 | 0                                  |
| num_train_epochs       | 进行训练的epoch数                              | 0                                  |
| warmup_proportion      | 线性学习率热身的训练比例                       | 0.1                                |
| save_checkpoints_steps | 保存模型ckpt的频率                             | 1000                               |
| amp                    | 是否使用amp进行混合精度训练                    | False                              |

# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow1上完成Bert-NGC的训练与推理。

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
# Modify the YOUR_IMAGE_NAME and IMAGE_TAG according to your own environment.
# For instance,
# IMAGE_NAME=tensorflow1-1.14.0-x86_64-ubuntu18.04
# IMAGE_TAG=YOUR_IMAGE_TAG

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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/BERT_ngc` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd tensorflow_modelzoo/tensorflow2/built-in/tools/record_time
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
cd tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/BERT_ngc

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=bert_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../
```

**b)创建并启动容器**

上一步成功运行后，本地便生成了一个名为`bert_image`的镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为bert_image
# 2. 运行run_docker.sh
bash run_docker.sh
```

### 4.2.2 **数据集准备**
本仓库使用的训练数据集是`squad-v1.1`数据集，可从[此处](https://github.com/google-research/bert#squad-11)下载。下载至本地后，数据集的存放路径可参考下方的目录结构：
```bash
/data/tensorflow/training/datasets/Bert/SQuAD
├──train-v1.1.json
├──dev-v1.1.json
└──evaluate-v1.1.py
```

### 4.2.3 **预训练模型准备**
预训练模型可从[此处](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md#checkpoints)下载，本仓库使用的是`cased_L-12_H-768_A-12`预训练模型。下载至本地后，参考如下命令解压即可：
```bash
tar -xvf cased_L-12_H-768_A-12.tar.gz
```
解压后，预训练模型的存放路径可参考下方的的目录结构：
```bash
/data/tensorflow/training/datasets/Bert/cased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
└── vocab.txt
```

### 4.2.4 **环境变量修改**

完成上述准备后，根据数据集与预训练模型的实际路径修改`env.sh`内的`MODEL_DIR`与`SQUAD_DIR`的值。

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

进入`run_scripts/`，该目录内提供了用于from_scratch的训练脚本。

| Models   | Framework   | Supported MLU | Data Precision | Cards | Run                                   |
|----------|-------------|---------------|----------------|-------|---------------------------------------|
| Bert-NGC | TensorFlow1 | MLU370-X8     | FP32           | 1     | bash Bert_Float32_2E_1MLU.sh          |
| Bert-NGC | TensorFlow1 | MLU370-X8     | FP32           | 8     | bash Horovod_Bert_Float32_2E_8MLUs.sh |
| Bert-NGC | TensorFlow1 | MLU370-X8     | AMP            | 8     | bash Horovod_Bert_APM_2E_8MLUs.sh     |

根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`train_file`，`train_batch_size`, `np`等，按照上述命令即可开始from_scratch训练。

若您想基于其他预训练模型进行finetune训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Bert_Float32_2E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，假设加载Bert-NGC的model.ckpt-xxxx进行finetune训练，训练1000 steps
# 请把预训练的模型及其配置文件放到`PATH_TO_CKPT`目录下，然后设置train_steps的值为1000

#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/bert_model_${timestamp}"
checkpoint_dir=PATH_TO_CKPT
pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 bert_trainer.py \
  --hvd_device=mlu \
  --train_steps=1000 \
  --bert_config_file="${checkpoint_dir}/bert_config.json" \
  --vocab_file="${checkpoint_dir}/vocab.txt" \
  --output_dir=$model_dir \
  --dllog_path="./bert_dllog.json" \
  --train_file="${SQUAD_DIR}/train-v1.1.json" \
  --predict_file="${SQUAD_DIR}/dev-v1.1.json" \
  --eval_script="${SQUAD_DIR}/evaluate-v1.1.py" \
  --init_checkpoint="${checkpoint_dir}/model.ckpt-xxxx"  \
  --do_lower_case=False \
  --max_seq_length=384 \
  --doc_stride=128 \
  --do_train=True \
  --do_predict=True \
  --train_batch_size=8 \
  --use_horovod=True \
  --use_profiler=False \
  --use_performance=False \
  --inter_op_threads=0 \
  --intra_op_threads=0 \
  --num_train_epochs=2 \
  --warmup_proportion=0.1 \
  --save_checkpoints_steps=1000 \
  --amp=False 

popd
```
**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与from_scratch得到该预训练模型的超参一致，否则无法正常训练。

### 4.3.2 **一键执行推理脚本**
本仓库提供了推理Bert-NGC网络的脚本：`run_scripts/infer_run_eager_bert.sh`，需要用户手动传入模型参数，默认以`tensorflow_modelzoo/tensorflow/built-in/NaturalLanguageProcessing/BERT_ngc`为当前目录。具体参见`3. 默认参数配置`，具体示例如下：
```bash
cd run_scripts
bash infer_run_eager_networks.sh PATH_TO_CKPT
```

# 5.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 6.Release_Notes
@TODO
