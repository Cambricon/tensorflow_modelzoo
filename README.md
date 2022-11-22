# TensorFlow ModelZoo 

## 介绍

TensorFlow是时下最流行的AI框架，寒武纪对其进行了定制化开发，新增了对寒武纪加速板卡及寒武纪AI软件栈的支持，通常称之为Cambricon TensorFlow。相比于原生TensorFlow，用户基本不用做任何代码改动即可快速地将AI模型迁移至Cambricon TensorFlow上。

针对CV分类、检测、分割、NLP、语音等场景常用的各类经典和前沿的AI模型，本仓库展示了如何对其进行适配，使其可运行在Cambricon TensorFlow上。开发者在进行其他AI应用移植时可参考本仓库。


## 网络支持列表和链接

CV：

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [resnet50_tf_model_official_v2.8.0](tensorflow2/built-in/Classification/resnet50_tf_model_official_v2.8.0) | TensorFlow2|FP32&&AMP|YES| TFMM/CNNL |
| [vgg16](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP |YES| TFMM/CNNL | 
| [vgg19](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL| 
| [resnet18](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL | 
| [resnet50](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL | 
| [resnet101](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL | 
| [densenet201](tensorflow2/built-in/Classification/common_networks) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL | 
| [centernet](tensorflow2/built-in/Detection/centernet) | TensorFlow2|FP32&&AMP | YES|TFMM/CNNL| 


NLP:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [transformer](tensorflow2/built-in/NaturalLanguageProcessing/transformer) | TensorFlow2|FP32 | YES | TFMM/CNNL |
| [google_bert](tensorflow2/built-in/NaturalLanguageProcessing/google_bert) | TensorFlow2|FP32&&AMP | YES | TFMM/CNNL |

Recommendation:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [DLRM](tensorflow2/built-in/Recommendation/DLRM) | TensorFlow2|FP32&&AMP | YES | CNNL|

Speech:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [Conformer](tensorflow2/built-in/ASR/Conformer) | TensorFlow2|FP32&&AMP|YES| CNNL |
| [LPCNet](tensorflow2/built-in/TTS/LPCNet) | TensorFlow2|FP32&&AMP|YES| CNNL |




## issues/wiki/forum 跳转链接

## contrib 指引和链接

## LICENSE

TensorFlow ModelZoo  的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

TensorFlow ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 TensorFlow ModelZoo ，TensorFlow ModelZoo  也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 TensorFlow ModelZoo上，或者您希望更新 TensorFlow ModelZoo中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。

