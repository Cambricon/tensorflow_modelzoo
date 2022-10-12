**Conformer (TensorFlow2)**

本仓库是在MLU上基于TensorFlow2框架实现的Conformer网络，支持训练与推理。


------------

**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.模型支持情况](#2.支持情况)
  * [2.1训练模型支持情况](#2.1训练模型支持情况)
  * [2.2推理模型支持情况](#2.2推理模型支持情况)
* [3.默认参数说明](#3.默认参数说明)
  * [3.1模型训练参数说明](#3.1模型训练参数说明)
  * [3.2模型推理参数说明](#3.2模型推理参数说明)  
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
Conformer是一个结合了CNN与Transformer的语音识别网络模型，输入是音频文件，输出是识别的文字结果。原始论文为[Conformer](https://arxiv.org/pdf/2005.08100.pdf).

Conformer网络的TensorFlow原生代码实现可参考：[这里](https://github.com/TensorSpeech/TensorFlowASR/tree/main/examples/conformer)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Conformer | TensorFlow2  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**


Models  | Framework  | Supported MLU   | Supported Data Precision  |  Eager Support
----- | ----- | ----- | -----  | ----- |
Conformer | TensorFlow2  | MLU370-S4/X4/X8  | FP16/FP32  | Jit&Eager 

注意，此处`Jit`表示使用`TFMM`的方式进行推理，即是用`TensorFlow2-MagicMind`作为底层实现后端进行推理。

# 3. 默认参数配置

## 3.1 **模型训练参数说明**


Conformer模型的训练参数存在于`conformer_train.py`内，同时受到`config.yml`及`run_scripts/`内的shell脚本的共同影响。

（1）`run_scripts`/内的shell脚本涉及到的常用参数及含义如下表所示：

<details>
<summary>展开查看</summary>


| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 训练的batch_size | 16   |
| data_dir | 音频文件的路径，用于训练。| your_path/LibriSpeech |
| steps | 不为0时，表示用户自主设定的每个epoch的步数。 | 0 |
| pretrained | 指向预训练模型路径 | None|
| mxp | 是否使用amp进行混合精度训练 | False |
| use_horovod | 是否使用horovod进行分布式训练 | True |
| use_gpu| 是否使用gpu进行训练 | False |
| use_profiler| 是否支持tensorboard，若为True则表示| False |
| use_performance | 是否开启性能测试，若为True则表示开启，训练结束后可在summary/summary.json内读出throughput与e2e| False |

</details>


（2）在本仓库根目录内还有`config.yml`，该配置文件内存放了大量的模型配置选项，如`epochs`，`checkpoint`文件保存路径等。需要注意的是，由于`config.yml`内的某些参数选项与（1）中的参数有重合，因此实际起作用的参数选项需根据`conformer_train.py`内的代码来判断。例如`epochs`，`config.yml`及`run_scripts/`内的shell脚本均可控制该参数，代码内对该参数的处理如下：
```bash
epochs=1 if FLAGS.steps > 0 else config.learning_config.running_config.num_epochs
# 若shell脚本中的steps参数非0，则epochs为1，否则epochs的值由config.yml内的[running_config]的num_epochs选项给出
```
其他类似的参数有`batch_size`。  

（3）还需特别注意的参数：`steps_per_epoch`，该参数的取值受到的因素较多，如下方代码所示：
```bash
steps_per_epoch = FLAGS.steps if FLAGS.steps > 0 else train_dataset_total_steps
# 若shell脚本内的steps非0，则steps_per_epoch由shell脚本内的steps给出
# 若shell脚本内的steps为0，则steps_per_epoch由train_dataset_total_steps给出
# train_dataset_total_steps的取值由训练数据集总数与batch_size相除给出
```

  
## 3.2 ** 模型推理默认参数说明**
<span id= "jump1"></span>
### 3.2.1 ** 模型推理常用参数说明**

推理的公共参数(如量化相关的参数)都在`../../tools/infer_flags.py`内，非推理独有的参数则在`model/resnet.py`内的common函数。程序运行时会解析并读取该脚本内的所有参数。
大部分参数提供了默认值，这些参数的详细含义将在稍后给出。
我们根据常用的参数组合，在`run_scripts/`下提供了若干个常用的脚本，如`infer_run_eager_fp32_bsz_4.sh`，`infer_run_jit_fp32_bsz_4.sh`，在使用这些脚本之前，您需要根据当前环境修改如下常用参数：
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

### 3.2.2 ** 模型推理其他参数说明**

除了上述常用参数之外，`infer_run_eager_fp32_bsz_4.sh` （infer_run_jit_fp32_bsz_4.sh与之类似）还有如下参数可供使用与修改：
```bash
native_savedmodel_dir#该参数表示原生TensorFlow Keras原始模型（savedmodel格式）的存放路径，需要传入绝对路径
converted_savedmodel_dir#该参数表示转换为TFMM模型（savemodel格式）的存放路径，需要传入绝对路径
imagenet_label_file#imagenet标签文件的绝对路径，通常是${data_dir}/val.txt
result_path#推理结果保存路径，需要为绝对路径
net_name#网络名称，本仓库内为Conformer
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
若要在脚本中使用过更多的参数，则需在`run_scripts/*.sh`脚本中新增对应的变量，再参照例如`quant_precision`的方式传入`resnet_infer.py`.
  
  
# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成Conformer的训练与推理。
## 4.1 **环境依赖项检查**
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

在容器内使用 `git clone` 下载本仓库代码并进入`tensorflow_modelzoo/tensorflow2/built-in/ASR/Conformer` 目录。

**d)安装模型依赖项**

```bash
# 安装requirements中的依赖库
pip install -r requirements.txt
pip install tensorflow-io~=0.18.0 --no-deps
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
cd tensorflow_modelzoo/tensorflow2/built-in/ASR/Conformer

# 4. 参考 前文 (1)基于base docker image的容器环境搭建 a)小节，获取基础镜像，假设镜像名字为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 5. 修改DOCKERFILE内的FROM_IMAGE_NAME的值为cambricon_tensorflow2:vX.Y.Z-x86_64-ubuntu18.04

# 6. 开始基于DOCKERFILE构建镜像
export IMAGE_NAME=conformer_image
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../../

```

**b)创建并启动容器**  

上一步成功运行后，本地便生成了一个名为`conformer_image`的docker镜像，后续即可基于该镜像创建容器。
```bash
# 1. 参考前文(1)基于base docker image的容器环境搭建 b) 小节，修改run_docker.sh 内的IMAGE_NAME为conformer_image
# 2. 运行run_docker.sh
bash run_docker.sh

```


### 4.2.2 **数据集准备**
本仓库使用的训练数据集是[LibriSpeech](https://www.openslr.org/12)。下载并解压到本地（数据集路径由`data_dir`参数指出）后，需保证与如下目录结构一致：
```bash
.
├── BOOKS.TXT
├── CHAPTERS.TXT
├── LICENSE.TXT
├── README.TXT
├── SPEAKERS.TXT
├── dev-clean
├── dev-other
├── test-clean
├── train-clean-100
├── train-clean-360
└── train-other-500
```

随后，还需根据`LibriSpeech`的本地路径修改`env.sh`内的`DATA_DIR`的值。

## 4.3 **运行Run脚本**

### 4.3.1 **一键执行训练脚本**
`run_scripts/`目录下提供了from_scratch的训练脚本。


Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Conformer| TensorFlow2  | MLU370-X8  | Float32  |  8 |Horovod_Conformer_Float32_50E_8MLUs.sh
Conformer  | TensorFlow2  | MLU370-X8  | AMP | 8  |Horovod_Conformer_AMP_50E_8MLUs.sh
Conformer  | TensorFlow2  | MLU370-X8  | Float32 | 1 |Conformer_Float32_50E_1MLU.sh


根据您的实际环境与需求，修改脚本内数据集的路径（`env.sh`内的`DATA_DIR`）及其他参数的值，如`batch_size`，`steps`，`mxp`等，按照如下命令即可开始from_scratch的分布式训练：
```bash
bash Horovod_Conformer_Float32_50E_8MLUs.sh
```
若您想基于已有的预训练模型进行训练，则可参考如下命令，修改脚本内的参数（以`Horovod_Conformer_Float32_50E_8MLUs.sh`为例）：
```bash
# fp32精度下，使用8卡MLU370-X8，
# 加载第50个epoch的checkpoint文件进行finetune训练1000代
# （finetune 1个epoch，每个epoch迭代1000 step）：
horovodrun -np 8 python conformer_train.py \
 --pretrained=YOUR_CKPT_PATH/50.h5 \
 --data_dir=YOUR_PATH/LibriSpeech \
 --batch_size=4 --num_workers=1 \
 --use_gpu=False --skip_eval=False \
 --steps=1000 --use_horovod=True \
 --use_profiler=True --mxp=False \
 --use_performance=False
```
**注意**：使用预训练模型进行finetune训练时，`batch_size`，`np`，`mxp`需与from_scratch得到该预训练模型的参数一致，否则无法正常训练。

### 4.3.2 ** 一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种参数配置的脚本：`run_scripts/multi_infer_run.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`quant_precision`，完成修改后，按照如下命令运行即可分别以不同的参数与推理模式（eager/jit）推理。
```bash
bash multi_infer_run.sh

```

目前支持的精度类型与推理模式组合以及运行环境如下所示：

Models  | Framework  | MLU   | Data Precision  |  Jit/Eager 
----- | ----- | ----- | ----- | ----- | 
ResNet50  | TensorFlow  | MLU370  | FP32  |  Eager 
ResNet50  | TensorFlow  | MLU370  | FP16  |  Eager
ResNet50  | TensorFlow  | MLU370  | FP32  |  Jit 
ResNet50  | TensorFlow  | MLU370  | FP16  | Jit  


# 5. **结果展示**

## 5.1 **训练结果**

**Training accuracy results: MLU370-X8**

Conformer的训练精度可由模型在测试集上的`wer`与`cer`表征。在`test_clean`数据集上，以fp32精度类型训练的模型精度如下：

 MLUs |Batch_size(Train)/Batch_size(Test)  | Epochs  |Greedy_wer  |   Greedy_cer
 ----- | ----- | ----- | -----  |-----  |
 8  |4/1  | 50|17.36%|8.02%


**Training performance results: MLU370-X8**

在运行`conformer_train.py`时候传入`--use_performance=True` 参数。
以下性能结果基于cambricon-tensorflow2(v1.12.1)取得。由于Conformer中能以fp16精度运行的算子较少，大量的算子仍以fp32精度运行，因此，数据类型转换（fp32转fp16）导致的耗时增加与算子与算子以fp16精度运行导致的耗时减少基本持平，从性能表现来看，便会出现混合精度训练的fps只比fp32精度训练的fps略高的情况。

Models   | MLUs |Batch Size  | Throughput(FP32)  | Throughput(Mixed Precision)  
----- | ----- | ----- | ----- | ----- | 
Conformer  | 1  |4| 6.00  | 5.97  
Conformer  | 4  |16| 18.70  | 19.50  
Conformer  | 8  |32| 33.89  | 34.57  





## 5.2  ** 推理结果**


###  Infering  results: MLU370-X4

Models | mode   | precision  | batch_size|top1/top5  | hardware_fps  
----- | ----- | ----- | ----- | ----- | ----- 
  ResNet50 |jit   | fp16  |32   | 0.68/0.88   | 4390.52  
  ResNet50 |jit   | fp32  |32   | 0.68/0.88   | 4701.54  
  ResNet50 |jit   | fp16  |64   | 0.68/0.88   | 5133.71  
  ResNet50 |jit   | fp32  |64   | 0.68/0.88   | 4701.54  
  ResNet50 |eager   | fp16  |32   | 0.68/0.88   | 2578.47  
  ResNet50 |eager   | fp32  |32   | 0.68/0.88   | 2587.58  
  ResNet50 |eager   | fp16  |64   | 0.68/0.88   | 6603.60  
  ResNet50 |eager   | fp32  |64   | 0.68/0.88   | 6825.25 



# 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

# 7.Release_Notes
@TODO


