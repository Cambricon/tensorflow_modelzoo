# Preparation

## Download COCO weights as TF checkpoint
```sh
cd checkpoint
wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
tar -xvf yolov3_coco.tar.gz
cd ..
```

# Training
1. change hyper parameters in core/config.py, such as initial learning rate, warmup epochs, first stage training epochs, etc.
2. change coco dataset prefix in data/dataset/coco17_train.txt and data/dataset/coco17_val.txt
2. start training

```sh
python train.py
```

# Inference
1. change test checkpoint path in core/config.py
2. change line 79 in evaluate_coco.py to fit your coco annotation file path.
2. start inference

```sh
python evaluate_coco.py
```
