# TensorFlow ModelZoo

## 介绍

TensorFlow是时下最流行的AI框架，寒武纪对其进行了定制化开发，新增了对寒武纪加速板卡及寒武纪AI软件栈的支持，通常称之为Cambricon TensorFlow。相比于原生TensorFlow，用户基本不用做任何代码改动即可快速地将AI模型迁移至Cambricon TensorFlow上。

针对CV分类、检测、分割、NLP、语音等场景常用的各类经典和前沿的AI模型，本仓库展示了如何对其进行适配，使其可运行在Cambricon TensorFlow上。开发者在进行其他AI应用移植时可参考本仓库。


## 网络支持列表和链接

CV：

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode | XLA Support |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [vgg19](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|CNNL|YES|
| [resnet50](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|CNNL |YES|
| [resnet101](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|CNNL |YES|
| [densenet201](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|CNNL |YES|
| [swin-transformer](tensorflow2/built-in/Classification/swin-transformer/) | TensorFlow2|FP32&&AMP |YES| CNNL |YES|
| [centernet](tensorflow2/built-in/Detection/centernet) | TensorFlow2|FP32&&AMP | YES|CNNL|NO|
| [ResNet50](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP | YES|CNNL |NO|
| [ResNet101](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP | YES|CNNL |NO|
| [DenseNet201](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP | YES|CNNL |NO|
| [Vgg19](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [InceptionV3](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [MobilenetV2](tensorflow/built-in/Classification/common_networks) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [SSD](tensorflow/built-in/Detection/SSD) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [YOLOv3](tensorflow/built-in/Detection/YOLOv3) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [retinanet](tensorflow/built-in/Detection/retinanet) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [UNet_3D_Medical](tensorflow/built-in/Segmentation/UNet_3D_Medical) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|
| [UNet_Industrial](tensorflow/built-in/Segmentation/UNet_Industrial) | TensorFlow1|FP32&&AMP |YES| CNNL |NO|

Graph Convolutional Network

| MODELS                                         | FRAMEWORK   | Train Mode | Distributed Train | Infer  Mode | XLA Support |
|------------------------------------------------|-------------|------------|-------------------|-------------|-------------|
| [GraphSAGE](tensorflow/built-in/GCN/GraphSAGE) | Tensorflow1 | FP32&&AMP  | NO                | CNNL        |NO           |

NLP:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode | XLA Support |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [transformer](tensorflow2/built-in/NaturalLanguageProcessing/transformer) | TensorFlow2|FP32 | YES | CNNL | YES |
| [google_bert](tensorflow2/built-in/NaturalLanguageProcessing/google_bert) | TensorFlow2|FP32&&AMP | YES | CNNL | YES |
| [tf_models_bert](tensorflow2/built-in/NaturalLanguageProcessing/tf_models_bert) | TensorFlow2|FP32 | YES | CNNL | NO |
| [BERT_ngc](tensorflow/built-in/NaturalLanguageProcessing/BERT_ngc) | TensorFlow1|FP32&&AMP | YES | CNNL | NO |
| [BERT_CRF](tensorflow/built-in/NaturalLanguageProcessing/bert/bert_crf) | TensorFlow1|FP32&&AMP | YES | CNNL | NO |
| [transformer_estimator](tensorflow/built-in/NaturalLanguageProcessing/Transformer/transformer_estimator/) | TensorFlow1|FP32/AMP | YES | CNNL | NO |
| [google_bert(tf1)](tensorflow/built-in/NaturalLanguageProcessing/google_bert) | TensorFlow1|FP32&&AMP | YES | CNNL | NO |

Recommendation:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode | XLA Support |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [DLRM](tensorflow2/built-in/Recommendation/DLRM) | TensorFlow2|FP32&&AMP | YES | CNNL| YES |
| [DeepFM](tensorflow/built-in/Recommendation/DeepFM) | TensorFlow1|FP32&&AMP | YES | CNNL| NO |

Speech:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode | XLA Support |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [Conformer](tensorflow2/built-in/ASR/Conformer) | TensorFlow2|FP32&&AMP|YES| CNNL | No |
| [LPCNet](tensorflow2/built-in/TTS/LPCNet) | TensorFlow2|FP32&&AMP|YES| CNNL | No |
| [Tacotron2](tensorflow/built-in/TTS/Tacotron-2) | TensorFlow1|FP32&&AMP|YES| CNNL | No |


## issues/wiki/forum 跳转链接

## contrib 指引和链接
## Change Log
---------- v1.6.0 2023年9月22日 ---------- 

- 新增了tf1 resnet50_v1.5, 且tf modelzoo v1.6.0与tf v1.20.0版本相对应

---------- v1.5.1 2023年7月12日 ---------- 

- 修复了dlrm的一个 rank 的bug，更新了change log.

---------- v1.5.0 2023年7月12日 ---------- 

- 修复了一些bug，tf modelzoo v1.5.0与tf v1.19.0版本相对应

---------- v1.4 2023年5月17日 ---------- 

- 向 tensorflow2 目录内添加了 tf_models_bert 网络 

---------- 2023年5月9日 ----------

- 支持网络列表删除了tensorflow1中的Resnet18,Alexnet,Vgg16,Inceptionv2,Resnet50-1.5，删除了tensorflow2中的resnet50_tf_model_official_v2.8.0,vgg16,resnet18。
- 添加了网络是否支持XLA模式的说明。

---------- v1.3 2023年3月31日 ---------- 

将tensorflow2/built-in 内使用 TFMM 进行推理的网络改为使用 CNNL 进行推理，涉及到的网络如下：
[vgg16](tensorflow2/built-in/Classification/common_networks)
[vgg19](tensorflow2/built-in/Classification/common_networks)
[resnet18](tensorflow2/built-in/Classification/common_networks)
[resnet50](tensorflow2/built-in/Classification/common_networks)
[resnet101](tensorflow2/built-in/Classification/common_networks)
[densenet201](tensorflow2/built-in/Classification/common_networks)
[centernet](tensorflow2/built-in/Detection/centernet)
[transformer](tensorflow2/built-in/NaturalLanguageProcessing/transformer)
[google_bert](tensorflow2/built-in/NaturalLanguageProcessing/google_bert)


## LICENSE

TensorFlow ModelZoo  的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

TensorFlow ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 TensorFlow ModelZoo ，TensorFlow ModelZoo  也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 TensorFlow ModelZoo上，或者您希望更新 TensorFlow ModelZoo中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。

