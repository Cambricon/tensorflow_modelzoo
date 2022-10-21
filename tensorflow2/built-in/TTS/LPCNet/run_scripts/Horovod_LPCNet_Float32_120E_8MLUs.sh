#!/bin/bash
set -e
set -x
cur_path=$(pwd)
work_dir="${cur_path}/../"
pushd $work_dir
source env.sh
horovodrun -np 8  python lpcnet_train.py \
 --num_mlus=1 \
 --num_gpus=0 \
 --batch_size=16 \
 --epochs=120 \
 --steps_per_epoch=1000 \
 --features="${DATA_DIR}/features.f32" \
 --data="${DATA_DIR}/data.u8" \
 --output=mlu_model \
 --model_dir=mlu_model \
 --start_epoch=0 \
 --enable_tensorboard=False \
 --use_amp=False \
 --use_performance=False \
 --use_horovod=True
popd
