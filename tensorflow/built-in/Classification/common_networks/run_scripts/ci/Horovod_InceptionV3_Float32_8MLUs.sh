#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
model_dir="${work_dir}/inceptionv3_model_float"

pushd "${work_dir}"

source env.sh

horovodrun -np 4 python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=inceptionv3 \
    --mode=train_and_eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=64 \
    --epochs=1 \
    --start_epoch=0 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=True \
    --skip_eval=True \
    --finetune_steps=10 \
    --finetune_checkpoint="" \
    --validation_steps=0 \
    --inter_op_threads=0 \
    --intra_op_threads=0 \
    --use_dummy_synthetic_data=False
popd


