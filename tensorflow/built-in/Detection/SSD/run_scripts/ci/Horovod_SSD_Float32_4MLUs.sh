#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/ssd_model_float"

pushd "${work_dir}"

source env.sh

horovodrun -np 4 python3 model_main.py \
    --batch_size=32 \
    --pipeline_config_path="models/configs/ssd320_full_4mlus.config" \
    --num_steps=10 \
    --do_train=True \
    --fine_tune_checkpoint="" \
    --fine_tune_checkpoint_type="classification" \
    --model_dir=$model_dir \
    --use_horovod=True \
    --hvd_device=mlu \
    --use_amp=False \
    --use_performance=False \
    --use_profiler=False
popd


