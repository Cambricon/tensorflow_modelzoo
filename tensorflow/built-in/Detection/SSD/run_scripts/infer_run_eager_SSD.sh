#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."

pushd "${work_dir}"

source env.sh

python3 model_main.py \
    --pipeline_config_path="models/configs/ssd320_full_1mlus.config" \
    --do_train=False \
    --model_dir="PATH_TRAINING_MODEL_DIR" \
    --use_horovod=False \
    --use_performance=False \
popd


