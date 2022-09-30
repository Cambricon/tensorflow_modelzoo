- [Precheckin 测例添加说明](#precheckin-测例添加说明)
- [Precheckin 触发说明](#precheckin-触发说明)
  - [全量测试](#全量测试)
  - [部分测试](#部分测试)
- [ci_pipeline.sh 脚本使用示例](#ci_pipelinesh-脚本使用示例)
## Precheckin 测例添加说明

相关测例添加在`ci/cases`目录中，在`precheckin_tf2.json`中添加tensorflow2相关的测例，
在`precheckin_tf1.json`中添加tensorflow1相关的测例，测例格式说明如下。

```json
"网络名称": {
    "directory": "网络相关代码所在目录", // 如果网络所在目录有requirements.txt，会自动pip install。
    "testcases":     [
        "echo testcase1", // 在该list中填入测例的命令, 每个命令会被封装为一个pytest
        "echo testcase2"
    ]
}
```

以`tensorflow2`的`resnet50_cmmc`为例，

模型所在目录在`tensorflow2/built-in/Classification/resnet50_cmcc`

执行单卡的测例的命令为

```bash
python3 -m resnet_trainer \
        --model_dir=./resnet50_cmcc_model \
        --data_dir=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord/ \
        --num_mlus=1 \
        --num_gpus=0 \
        --distribution_strategy=off \
        --batch_size=128 \
        --train_steps=10
```

则在`ci/cases/precheckin_tf2.json`中添加如下信息。

```json
    "resnet50_cmcc": {
        "directory": "tensorflow2/built-in/Classification/resnet50_cmcc",
        "testcases": [
            "python3 -m resnet_trainer --model_dir=./resnet50_cmcc_model --data_dir=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord/ --num_mlus=1 --num_gpus=0 --distribution_strategy=off --batch_size=128 --train_steps=10"
        ]
    }
```

## Precheckin 触发说明

### 全量测试

测试TensorFlow1和TensorFlow2的所有的模型，在mr中评论`rebuild`即触发流水。

### 部分测试

以只测试TensorFlow2的resnet50_cmcc和vgg16为例，在mr中评论`rebuild tf2 resnet50_cmcc,vgg16`即触发流水。


## ci_pipeline.sh 脚本使用示例

需要在项目的根目录下执行

```bash
# 测试 ci/cases/precheckin_tf2.json 中所有的测例
bash ci/ci_pipeline.sh precheckin_tf2
# 测试 ci/cases/precheckin_tf2.json 中指定的模型测例
bash ci/ci_pipeline.sh precheckin_tf2 resnet50_cmcc,VGG16
```

