#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/vgg16_model_${timestamp}"

pushd "${work_dir}"

source env.sh

python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=vgg16 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=64 \
    --epochs=1 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=True \
    --run_eagerly=False \
    --skip_eval=False \
    --finetune_steps=10 \
    --finetune_checkpoint="" \
    --enable_tensorboard=False \
    --datasets_num_private_threads=0
popd


