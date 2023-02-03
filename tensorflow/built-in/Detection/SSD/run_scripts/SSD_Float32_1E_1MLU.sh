#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/ssd_model_${timestamp}"

pushd "${work_dir}"

source env.sh

python3 model_main.py \
    --batch_size=32 \
    --pipeline_config_path="models/configs/ssd320_full_1mlus.config" \
    --num_steps=100000 \
    --do_train=True \
    --fine_tune_checkpoint="" \
    --fine_tune_checkpoint_type="classification" \
    --model_dir=$model_dir \
    --use_horovod=False \
    --hvd_device=mlu \
    --use_amp=False \
    --use_performance=False \
    --use_profiler=False
popd


