#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
model_dir="${work_dir}/yolov3_model_single"

pushd "${work_dir}"

source env.sh
ANNOT_PATH="${work_dir}/models/data/dataset/coco17_train.txt"
sed -i s:^0:${TRAIN_FILE_LIST_PATH}0:g ${ANNOT_PATH}

python3 train.py \
    --batch_size=8 \
    --first_stage_epochs=0 \
    --second_stage_epochs=1 \
    --start_epoch=0 \
    --ckpt_path="" \
    --finetune_step=10 \
    --output_dir=$model_dir \
    --use_horovod=False \
    --hvd_device=mlu \
    --use_amp=False \
    --use_performance=False \
    --use_profiler=False
popd


