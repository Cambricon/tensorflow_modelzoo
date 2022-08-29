**ResNet50 (TensorFlow)**

本仓库是在MLU上基于TensorFlow框架实现的ResNet50网络，支持训练与推理。

------------

**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.模型支持情况](#2.支持情况)
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
ResNet50网络是残差卷积神经网络，原始论文为[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。

ResNet50网络结构的代码实现可参考：[这里](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet)。
# 2. 模型支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50_CMCC | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision   | 






/Eager Support
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50_CMCC   | TensorFlow  | MLU370-S4/X4/X8  | FP16/FP32   | Jit&Eager
注意，此处`Jit`表示使用`TFMM`的方式进行推理，即是用`TensorFlow2-MagicMind`作为底层实现后端进行推理。

# 3. 默认参数配置

## 3.1 **模型训练默认参数配置**

| 参数 | 作用 | 默认值 |
|------|------|------|
| batch_size | 更改训练的batch_size | 256 |
| model_dir | 指向保存checkpoint的路径 | ./mlu_model |
| data_dir | 指向数据集的路径 |  |
| train_epochs | 更改训练的epoch数目 | 90 |
| base_learning_rate | 更改训练初始学习率 | 0.2 |
| use_profiler | 为True则开启tensorboard | False |
| use_amp | 控制是否使用amp进行混合精度训练 | False |
  
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

## 4.2 环境准备
### 4.2.1 **数据集准备**
该ResNet50脚本基于ImageNet2012训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
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


### 4.2.2 **容器环境准备**
- #### 基于base docker image安装
##### （1）导入镜像
下载Cambricon TensorFlow docker镜像
 docker load -i xxx.tar.gz
##### （2）启动容器
`run_docker.sh`示例如下，修改如下示例中的`YOUR_XXX`变量后再运行`bash run_docker.sh`即可启动容器

```bash 
#!/bin/bash
#below is a sample of run_docker.sh,
#modify the uppercase var according to your own environment.
docker run -it --name=YOUR_DOCKER_NAME  \ 
 --network=host --cap-add=sys_ptrace \ 
 -v /YOUR_HOST_PATH/:/YOUR_DOCKER_INSIDE_PATH \
 -v /usr/bin/cnmon:/usr/bin/cnmon \ 
 --device=/dev/cambricon_dev0:/dev/cambricon_dev0 \
 --device=/dev/cambricon_ctl  
 YOUR_DOCKER_IMAGE_NAME:YOUR_DOCKER_IMAGE_TAG  /bin/bash

```

##### （3）下载项目代码并安装依赖
```bash
#使用 git clone 下载本仓库代码并进入tensorflow_modelzoo/tensorflow2/buildin/Classification/resnet50 目录
pip install -r requirements.txt
```
## 4.3 **运行Run脚本**
以推理为例，进入`scripts`目录，并参考[模型推理默认参数说明](#jump1)，修改`scripts`内的参数，随后开始推理：
```
#运行eager模式的示例脚本
bash infer_run_eager_fp32_bsz_4.sh
#运行jit模式的示例脚本
bash infer_run_jit_fp32_bsz_4.sh
```

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 8  | Horovod_ResNet50_Float32_90E_8MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | AMP O1  | 8  | Horovod_ResNet50_AMP_90E_8MLUs.sh

### 4.3.2 **+ 一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种参数配置的脚本：`scripts/multi_infer_run.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`quant_precision`，完成修改后，按照如下命令运行即可分别以不同的参数与推理模式（eager/jit）推理。
```bash
bash multi_infer_run.sh

```

目前支持的精度类型与推理模式组合以及运行环境如下所示：

Models  | Framework  | MLU   | Data Precision  |  Jit/Eager |
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow  | MLU370  | FP32  |  Eager 
ResNet50  | TensorFlow  | MLU370  | FP16  |  Eager 
ResNet50  | TensorFlow  | MLU370  | FP32  |  Jit 
ResNet50  | TensorFlow  | MLU370  | FP16  | Jit  


# 5. **结果展示**

## 5.1 **训练结果**

** Training accuracy results: MLU370-X8**

Models  | Epochs  | Mixed Precision Top1   | FP32 Top1 
----- | ----- | ----- | ----- | 
ResNet50  | 61  | 74.32 | N/A
ResNet50  | 100  | 74.44 | 76.124

** Training performance results: MLU370-X8**

Models   | MLUs   | Throughput(FP32)  | Throughput(Mixed Precision)  |  FP32 Training Time(100E) | Mixed Precision Training Time(100E)
----- | ----- | ----- | ----- | ----- | -----
ResNet50  | 1  | 360.71  | 712.24  | N/A| N/A
ResNet50  | 4  | 1374.23  | 2604.65  | N/A| N/A
ResNet50  | 8  | 2735.88  | 5120  | N/A| N/A

## 5.2  **+ 推理结果**


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



