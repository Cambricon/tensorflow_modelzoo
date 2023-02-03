#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
model_dir="${work_dir}/retinanet_model_single"

pushd "${work_dir}"

source env.sh

python3 train.py \
    --strategy_type=off \
    --num_mlus=1 \
    --num_gpus=0 \
    --mode=train \
    --batch_size=8 \
    --init_learning_rate=0.01 \
    --learning_rate_levels=[0.001,0.0001] \
    --learning_rate_steps=[439920,586560] \
    --iterations_per_loop=14664 \
    --use_amp=False \
    --total_steps=10 \
    --training_file_pattern=$TRAIN_DATA_DIR \
    --eval_file_pattern=$EVAL_DATA_DIR \
    --val_json_file=$VAL_JSON_FILE \
    --model_dir=$model_dir \
    --use_horovod=False \
    --use_performance=False \
    --use_profiler=False
popd


