#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/vgg19_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 classifier.py \
    --dataset=imagenet \
    --model_type=vgg19 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=64 \
    --epochs=100 \
    --use_performance=False \
    --use_amp=True \
    --use_horovod=True \
    --run_eagerly=False \
    --skip_eval=False \
    --finetune_steps=0 \
    --finetune_checkpoint="" \
    --enable_tensorboard=False \
    --datasets_num_private_threads=0
popd


