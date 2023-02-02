#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/retinanet_model_${timestamp}"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 train.py \
    --strategy_type=off \
    --num_mlus=1 \
    --num_gpus=0 \
    --mode=train \
    --batch_size=8 \
    --init_learning_rate=0.08 \
    --learning_rate_levels=[0.008,0.0008] \
    --learning_rate_steps=[54990,73320] \
    --iterations_per_loop=1833 \
    --use_amp=True \
    --total_steps=91650 \
    --training_file_pattern=$TRAIN_DATA_DIR \
    --eval_file_pattern=$EVAL_DATA_DIR \
    --val_json_file=$VAL_JSON_FILE \
    --model_dir=$model_dir \
    --use_horovod=True \
    --use_performance=False \
    --use_profiler=False
popd


