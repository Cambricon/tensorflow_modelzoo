#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."

pushd "${work_dir}"

source env.sh

python3 eval.py \
    --eval_file_pattern=$EVAL_DATA_DIR \
    --val_json_file=$VAL_JSON_FILE \
    --checkpoint_path=PATH_TRAINING_MODEL_DIR \
popd
