**ResNet50 (TensorFlow)**
本仓库是在MLU上基于TensorFlow框架实现的ResNet50网络，支持训练与推理。

------------




**目录 (Table of Contents)**
* [1.模型概述](#1.模型概述)
* [2.支持情况](#2.支持情况)
* [3.默认参数配置](#3.默认参数配置)
  * [3.1模型训练默认参数配置](#3.1模型训练默认参数配置)
  * [3.2模型推理默认参数配置](#3.2模型推理默认参数配置)
  * [3.3运行Run脚本](#3.3运行Run脚本)
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

ResNet50网络结构的代码实现可参考：[这里](https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/resnet.py#L440-L459)。
# 2. 支持情况
## 2.1 **训练模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50-v1.5  | TensorFlow  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested

## 2.2 **推理模型支持情况**

Models  | Framework  | Supported MLU   | Supported Data Precision  | Quantification  | Jit/Eager Support
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50-v1.5  | TensorFlow  | MLU370-S4/X4  | FP16/FP32  | int8/int16 | Jit&Eager
# 3. 默认参数配置

## 3.1 **模型训练默认参数配置**

以下为ResNet50模型的默认参数配置：

### 3.1.1 Optimizer

模型默认优化器为SGD，以下为相关参数：

* Momentum: 0.9
* Learning Rate: 0.2 for batch size 64
* Learning rate schedule: Linear schedule
* Weight decay: 1e-4
* Label Smoothing: None
* Epoch: 90

### 3.1.2 Data Augmentation

模型使用了以下数据增强方法：

* 训练
  * Normolization
  * Crop image to 224*224
  * RandomHorizontalFlip
* 验证
  * Normolization
  * Crop image to 256*256
  * Center crop to 224*224
  
  ## 3.2 **+ 模型推理默认参数配置**
  <span id= "jump1"></span>
  推理的所有参数都在`config/resnet50_config.json`内，程序运行时会解析并读取该文件内的所有配置选项。通常来说，您需要根据当前环境修改如下参数：
  
```bash
eval.mode.val# 该参数可选值为"jit"或者"eager"，若使用"jit"则表示使用TF2MM推理，若使用“eager”则表示使用CNNL推理，默认为“jit”。
eval.model.native_savedmodel_dir#该参数表示原生TensorFlow Keras原始模型（savedmodel格式）的存放路径，需要传入绝对路径
eval.model.converted_savedmodel_dir#该参数表示转换为TF2MM模型（savemodel格式）的存放路径，需要传入绝对路径
eval.infer_dataset.batch_size#推理时的batch大小，默认值为1
eval.infer_dataset.infer_data_dir#imagenet验证数据集的绝对路径，图片格式需为JPEG
eval.infer_dataset.imagenet_label_file#imagenet标签文件的绝对路径
eval.result_path#推理结果保存路径，需要为绝对路径
-----------------------------------
#如下参数只在jit模式时被使用
eval.infer_dataset.calibration_data_dir#用于量化校准的数据的绝对路径，图片格式需为JPEG。
eval.mm_opt{
quant_precision#精度类型，可选值为[int8_fp16,int8_fp32,int16_fp16,int16_fp32,fp16,fp32]其中之一，默认为fp32。当精度为[int8_fp16,int8_fp32,int16_fp16,int16_fp32]其中之一时（混合精度），程序会使用eval.infer_dataset.calibration_data_dir内的数据对模型进行量化
quantize_algo#量化算法，可选值为['linear','eqnm']其中之一，默认值为'linear'
quantize_type#量化类型，可选值为['symm','asym']其中之一，即对称量化或非对称量化，默认值为'symm'
quantize_granularity#量化维度， 输入为 ['per_tensor', 'per_axis'] 其中之一，即指定逐tensor量化还是逐维度量化，默认为 'per_tensor'
enable_dim_range#对于输入batch维度可变的模型，启用设置可变维度范围能提升模型性能，默认为 Fasle
opt_config#TF2MM模型优化性能选项，目前支持的输入为 [conv_scale_fold,type64to32_conversion] 如果想需要设置多个，用逗号 ',' 隔开
}
eval.visible_devices#对于多卡用户，设置程序运行在哪张卡上，默认为 0
```
  
  
  
# 4.快速使用
下面将详细展示如何在 Cambricon TensorFlow2上完成ResNet50的训练与推理。
## 4.1 **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNTensorFlow >= 1.3.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO
## 4.2 环境准备
### 4.2.1 **数据集准备**
该ResNet50脚本基于ImageNet1K训练，数据集下载：[https://www.image-net.org/](https://www.image-net.org/)
本地数据集目录结构请与下方保持一致：
```
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── ...
├── train.txt
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── ...
└── val.txt
```
### 4.2.2 **容器环境准备**
- #### 基于base docker image安装
##### （1）导入镜像
```
##下载Cambricon TensorFlow docker镜像
docker load -i xxx.tar.gz
```
##### （2）启动容器

```bash
bash run_docker.sh
```

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
 THE_IMAGE_NAME:THE_IMAGE_TAG  /bin/bash

```

##### （3）下载项目代码并安装依赖

```bash
git clone http://gitlab.software.cambricon.com/wangle/tmp_slowfast_backup/-/commit/4ead60c1ddae1374a4a789073af68cc9287df25e
pip install -r requirements.txt
```
## 4.3 **运行Run脚本**
参考[前文所述](#jump1)修改`config/resnet50_config.json`内的配置选项，随后参考如下命令开始训练或推理。
```
bash train_run.sh config/resnet50_config.json
bash infer_run.sh config/resnet50_config.json
```

### 4.3.1 **一键执行训练脚本**

Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow  | MLU370-X8  | AMP O1  | 1  | bash MLU370X8_ResNet50_AMP_61E_2MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | FP32  | 2  | bash MLU370X8_ResNet50_FP32_90E_4MLUs.sh
ResNet50  | TensorFlow  | MLU370-X8  | AMP O1  | 8  | bash MLU370X8_ResNet50_AMP_90E_16MLUs.sh

### 4.3.2 **+ 一键执行推理脚本**
为了遍历多种输入规模与精度类型以及推理模式，本仓库还提供了一键执行多种配置文件的脚本：`multi_infer_run.sh`，您可根据自己的需求修改该脚本内的`batch_size`，`mode`，`precision`，完成修改后，按照如下命令运行：
```bash
bash multi_infer_run.sh config/resnet50_config.json

```

`multi_infer_run.sh`会对`resnet50_config.json`内的`eval.mode.val`,`eval.infer_dataset.batch_size`,`eval.mm_opt.quant_precision`覆盖并产生新的配置文件如`resnet50_config_bsz_8_int16_fp16_eager.json`，并根据新的配置文件调用`infer_run.sh`进行推理。
目前支持的精度类型与推理模式组合以及运行环境如下所示：

Models  | Framework  | MLU   | Data Precision  | Quant  | Jit/Eager |
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow  | MLU370-X4  | FP32  | no  | Eager |
ResNet50  | TensorFlow  | MLU370-X4  | FP16  | no  | Eager | 
ResNet50  | TensorFlow  | MLU370-X4  | FP32  | int8  | Jit | 
ResNet50  | TensorFlow  | MLU370-X4  | FP16  | int8  | Jit | 
ResNet50  | TensorFlow  | MLU370-X4  | FP32  | int16  | Jit | 
ResNet50  | TensorFlow  | MLU370-X4  | FP16  | int16 | Jit | 

# 5. **结果展示**

## 5.1 **训练结果**

** Training accuracy results: MLU370-X8**

Models  | Epochs  | Mixed Precision Top1   | FP32 Top1 
----- | ----- | ----- | ----- | 
ResNet50  | 61  | 74.32 | N/A
ResNet50  | 100  | 74.44 | 76.124

** Training performance results: MLU370-X8**

Models   | MLUs   | Throughput
(FP32)  | Throughput
(Mixed Precision)  |  FP32 Training Time
(100E) | Mixed Precision Training Time
(100E)
----- | ----- | ----- | ----- | ----- | -----|
ResNet50  | 1  | 360.71  | 712.24  | N/A| N/A
ResNet50  | 4  | 1374.23  | 2604.65  | N/A| N/A
ResNet50  | 8  | 2735.88  | 5120  | N/A| N/A

## 5.2  **+ 推理结果**


###  Infering  results: MLU370-X4

Models | mode   | precision  | batch_size|top1/top5  | hw_fps  
----- | ----- | ----- | ----- | ----- | ----- 
  ResNet50 |jit   | int8_fp16  |32   | 0.67/0.88  |  6877.97 
  ResNet50 |jit   | int8_fp32  |32   |  0.67/0.88 |  5523.20 
  ResNet50 |jit   | int16_fp16  |32   |  0.67/0.88 | 4876.88  
  ResNet50 |jit   | int16_fp32  |32   | 0.68/0.88   | 3827.62  
  ResNet50 |jit   | int8_fp16  |64   | 0.67/0.88  |  7933.52 
  ResNet50 |jit   | int8_fp32  |64   |  0.67/0.88 |  5878.80 
  ResNet50 |jit   | int16_fp16  |64   |  0.68/0.88 | 6001.92  
  ResNet50 |jit   | int16_fp32  |64   | 0.68/0.88   | 4701.54  
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

