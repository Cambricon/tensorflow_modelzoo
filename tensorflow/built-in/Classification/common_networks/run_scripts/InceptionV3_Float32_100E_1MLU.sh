#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/inceptionv3_model_${timestamp}"

pushd "${work_dir}"

source env.sh

python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=inceptionv3 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=64 \
    --epochs=100 \
    --start_epoch=0 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=False \
    --skip_eval=False \
    --finetune_steps=0 \
    --finetune_checkpoint="" \
    --validation_steps=0 \
    --inter_op_threads=0 \
    --intra_op_threads=0 \
    --use_dummy_synthetic_data=False
popd


