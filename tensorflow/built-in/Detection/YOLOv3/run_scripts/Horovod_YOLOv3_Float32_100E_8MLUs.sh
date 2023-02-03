#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/yolov3_model_${timestamp}"

pushd "${work_dir}"

source env.sh
ANNOT_PATH="${work_dir}/models/data/dataset/coco17_train.txt"
sed -i s:^0:${TRAIN_FILE_LIST_PATH}0:g ${ANNOT_PATH}

horovodrun -np 8 python3 train.py \
    --batch_size=8 \
    --first_stage_epochs=40 \
    --second_stage_epochs=60 \
    --start_epoch=1 \
    --ckpt_path="checkpoint/yolov3_coco_demo.ckpt" \
    --finetune_step=0 \
    --output_dir=$model_dir \
    --use_horovod=True \
    --hvd_device=mlu \
    --use_amp=False \
    --use_performance=False \
    --use_profiler=False
popd


