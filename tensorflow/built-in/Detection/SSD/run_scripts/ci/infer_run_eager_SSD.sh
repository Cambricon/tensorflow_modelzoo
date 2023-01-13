#!/bin/bash
set -e
set -x
cur_path=$(pwd)
work_dir="${cur_path}/../.."

pushd "${work_dir}"

source env.sh

python3 model_main.py \
    --pipeline_config_path="models/configs/ssd320_full_1mlus.config" \
    --do_train=False \
    --model_dir="run_scripts/ci/ssd_model_single" \
    --use_horovod=False \
    --use_performance=False \
popd


