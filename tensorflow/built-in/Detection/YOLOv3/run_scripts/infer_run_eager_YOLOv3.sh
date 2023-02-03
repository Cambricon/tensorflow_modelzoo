#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."

pushd "${work_dir}"

source env.sh
ANNOT_PATH="${work_dir}/models/data/dataset/coco17_train.txt"
sed -i s:^0:${VAL_FILE_LIST_PATH}0:g "${work_dir}/models/data/dataset/coco17_val.txt"

python3 evaluate_coco.py \
    --batch_size=2 \
    --image_number=4942 \
    --instances_path=$INSTANCES_PATH \
    --weight_file="YOUR_TRAINING_MODEL_FILE"
popd
