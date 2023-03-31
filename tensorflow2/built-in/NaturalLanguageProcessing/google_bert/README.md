**Bert (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的Bert网络，支持训练与推理。


------------

**目录 (Table of Contents)**
* [1.模型概述](#1-模型概述)
* [2.模型支持情况](#2-支持情况)
  * [2.1训练模型支持情况](#21-训练模型支持情况)
  * [2.2推理模型支持情况](#22-推理模型支持情况)
* [3.模型训练与推理参数说明](#3-模型训练与推理参数说明)
* [4.快速使用](#4-快速使用)
  * [4.1依赖项检查](#41-依赖项检查)
  * [4.2环境准备](#42-环境准备)
  * [4.3运行Run脚本](#43-运行Run脚本)
* [5.结果展示](#5-结果展示)
  * [5.1训练结果](#51-训练结果)
* [6.免责声明](#6-免责声明)
* [7.Release_Notes](#7-Release_Notes)


# 1. 模型概述
Bert是基于Transformer的一个网络模型，由多个Transformer的Encoder堆叠而成。Bert的创新之处主要在于在预训练时，通过使用Masked LM 和 Next Sentence Prediction两种方法捕捉了词语与句子级别的表征信息。在实际使用中，已完成预训练的Bert模型再加上特定的下游网络，如CRF，即可完成NLP的下游任务，如机器翻译，文本分类等。
原始论文为[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 。
本仓库使用Bert的开源预训练模型进行question-answering下游任务，需先finetune。其原生代码实现可参考[这里](https://github.com/google-research/bert)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes | XLA Support |
----- | ----- | ----- | ----- | ----- | ----- | ----- |
Bert | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested | Yes |

## 2.2 **推理模型支持情况**

|Models  | Framework  | Supported MLU   | Supported Data Precision   | Eager Support|
|----- | ----- | ----- | ----- | ----- |
|Bert   | TensorFlow2  | MLU370-S4/X4/X8  | FP32   | Eager|


# 3. 模型训练与推理参数说明


Bert模型训练与推理时涉及到的参数主要存在于`run_squad.py`。本仓库基于`squad-v1.1`数据集进行question-answering（以下简称为qa）任务。

（1）run_scripts/内的shell脚本涉及到的常用参数及含义如下表所示：
<details>
<summary>展开查看</summary>




| 参数 | 作用 | 默认值 |
|------|------|------|
| train_batch_size | 训练时每张MLU卡上的batch_size | 22   |
| predict_batch_size | 推理时每张MLU卡上的batch_size | 8   |
| max_seq_length| 文本序列最大长度 | 384   |
| num_train_epochs| 训练迭代次数 |2  |
| do_train | 是否进行训练 | True |
| do_predict | 训练结束后是否进行推理 | True |
| vocab_file | qa任务中词典文件 | your_path/uncased_L-12_H-768_A-12/vocab.txt |
| bert_config_file | qa任务bert网络结构设置，包含hidden_size,num_hidden_layers等|  your_path/uncased_L-12_H-768_A-12/bert_config.json |
| init_checkpoint | 初始预训练模型。若只进行推理，则需将该参数指向已训练完成的checkpoint文件 | your_path/uncased_L-12_H-768_A-12/bert_model.ckpt |
| train_file | 训练文件| your_path/SQuAD/train-v1.1.json |
| predict_file | 预测文件| your_path/SQuAD/dev-v1.1.json |
| eval_script | 用于训练结束时计算在验证集上的精度| your_path/SQuAD/evaluate-v1.1.py |
| output_dir | 模型输出路径，用于存放训练的checkpoint | mlu_model |
| use_amp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| enable_xla | 是否使能xla | False |
| use_performance | 是否开启性能测试，若为True则表示开启，训练结束后可在summary/summary.json内读出throughput与e2e| False |


</details>



（2）其他参数：

| 参数 | 作用 | 默认值 |
|------|------|------|
| save_checkpoints_steps| 每隔多少步保存一次checkpoint文件 | 1000 |
| use_profiler| 是否支持tensorboard，若为True则表示| False |
| finetune_steps| 通常用于控制finetune时的训练步数，默认为0，此时训练步数由batch_size，num_train_epochs，hvd.size等参数共同决定| 0 |
| learning_rate| 训练时的学习率| 1e-5 |



# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成Bert的训练与推理。
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

下载Cambricon TensorFlow2 镜像并参考如下命令加载镜像：
` docker load -i Your_Cambricon_TensorFlow2_Image.tar.gz`

**b)启动容器**

`run_docker.sh`示例如下，根据本地的镜像版本，修改如下示例中的`IMAGE_NAME`和`IMAGE_TAG`变量后再运行`bash run_docker.sh`即可启动容器。
```bash
#!/bin/bash
# Below is a sample of run_docker.sh.
# Modify the  YOUR_IMAGE_NAME and IMAGE_TAG	 according to your own environment.
# For instance,
# IMAGE_NAME=tensorflow2-1.12.1-x86_64-ubuntu18.04
# IMAGE_TAG=latest

IMAGE_NAME=YOUR_IMAGE_NAME
IMAGE_TAG=YOUR_IMAGE_TAG

export MY_CONTAINER="google_bert_tensorflow2_modelzoo"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER

if [ 0 -eq $num ];then
    xhost +
    docker run -it --name="${MY_CONTAINER}" \
     --net=host \
     --privileged=true \
     --cap-add=sys_ptrace \
     --shm-size="16g" \
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/NaturalLanguageProcessing/google_bert` 目录。

**d)安装模型依赖项**

```bash
#安装依赖库
pip install sentencepiece
# 安装性能测试工具(可选)
# 若不开启性能测试（use_performance为False），则无需安装。
cd ../../../../tools/record_time
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
cd tensorflow_modelzoo/tensorflow2/built-in/NaturalLanguageProcessing/google_bert

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

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
本仓库使用的训练数据集是`squad-v1.1`数据集，可从[此处](https://github.com/google-research/bert#squad-11)下载。下载至本地后的目录结构可参考下方：
```bash
/data/tensorflow/training/datasets/Bert/SQuAD
├──train-v1.1.json
├──dev-v1.1.json
└──evaluate-v1.1.py

```

### 4.2.3 **预训练模型准备**
预训练模型可从[此处](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1)下载，本仓库使用的是`uncased_L-12_H-768_A-12`预训练模型。下载至本地后，参考如下命令解压即可：
```bash
tar -xvf PRETRAINED_MODEL.tar.gz
```
解压后的目录结构如下：
```bash
uncased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── bert_model.ckpt.meta
└── vocab.txt

```

### 4.2.4 **环境变量修改**

完成上述准备后，还需根据数据集与预训练模型的实际路径修改`env.sh`内的`MODEL_DIR`与`SQUAD_DIR`的值。


## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**

进入`run_scripts/`，该目录内提供了用于finetune的训练脚本。


Models  | Framework  | Supported MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Bert| TensorFlow2  | MLU370-X8  | Float32  | 8  |bash Horovod_Bert_Float32_2E_8MLUs.sh
Bert  | TensorFlow2  | MLU370-X8  | AMP | 8  |bash Horovod_Bert_AMP_2E_8MLUs.sh
Bert  | TensorFlow2  | MLU370-X8  | Float32 | 1  |bash Bert_Float32_2E_1MLU.sh


根据您的实际环境与需求，修改脚本内数据集的路径及其他参数的值，如`train_file`，`train_batch_size`，`np`等，按照如下命令即可开始finetune训练：
```bash
bash Horovod_Bert_FP32_2E_8MLU.sh
```
若您想基于其他预训练模型进行finetune训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Bert_Float32_2E_8MLUs.sh`为例）：
```bash
# 使用8卡MLU370-X8，
# 加载已经训练了2个epoch的checkpoint文件进行finetune
# finetune1000 step

horovodrun -np 8 python run_squad.py \
 --use_horovod=True --use_amp=False \
 --finetune_steps=1000 --hvd_device=mlu \
 --max_seq_length=384 --num_train_epochs=2.0 \
 --doc_stride=128 --save_checkpoints_steps=1000 \
 --do_train=True --do_predict=True --learning_rate=1e-5 \
 --vocab_file=/YOUR_MODEL_PATH/uncased_L-12_H-768_A-12/vocab.txt \
 --bert_config_file=/YOUR_MODEL_PATH/uncased_L-12_H-768_A-12/bert_config.json \
 --init_checkpoint=/YOUR_CKPT_PATH/model.ckpt-497 \
 --train_file=/YOUR_DATA_PATH/SQuAD/train-v1.1.json \
 --predict_file=/YOUR_DATA_PATH/SQuAD/dev-v1.1.json \
 --eval_script="${dataset_path}/squad_evaluate_v1_1.py" \
 --train_batch_size=22 --use_profiler=False \
 --output_dir=mlu_model_finetune --use_performance=False
```

**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`use_amp`等超参需与得到该预训练模型的超参一致，否则无法正常训练。

### 4.3.1 **一键执行推理脚本**

进入`run_scripts/`，该目录内提供了用于推理的脚本。


Models  | Framework  | Supported MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Bert| TensorFlow2  | MLU370-S4/X4/X8  | Float32  | 1  |bash Infer_Bert_Float32_1MLU.sh




# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

Bert qa任务在squad数据集上的训练精度可由在测试集上取得的`f1`表征。

Models   | MLUs |Total Batch Size  | f1(FP32)  | f1(Mixed Precision)
----- | ----- | ----- | ----- | ----- |
Bert  | 8  |176| 89.13  | 88.04




# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO




