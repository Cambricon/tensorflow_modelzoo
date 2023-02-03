#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet18_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=resnet18 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=128 \
    --epochs=100 \
    --start_epoch=0 \
    --use_performance=False \
    --use_amp=True \
    --use_horovod=True \
    --skip_eval=False \
    --finetune_steps=0 \
    --finetune_checkpoint="" \
    --validation_steps=0 \
    --use_qat=False \
    --inter_op_threads=0 \
    --intra_op_threads=0 \
    --use_dummy_synthetic_data=False
popd


