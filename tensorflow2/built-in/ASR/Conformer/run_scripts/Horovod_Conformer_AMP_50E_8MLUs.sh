#!/bin/bash
set -e
set -x
export TF_MLU_THREAD_MODE="mlu_private"
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_MLU_THREAD_COUNT=8

cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
source env.sh
horovodrun -np 8 python conformer_train.py \
 --data_dir=$DATA_DIR \
 --batch_size=4 \
 --num_workers=1 \
 --use_gpu=False \
 --skip_eval=False \
 --use_horovod=True \
 --use_profiler=False \
 --use_performance=False \
 --mxp=True
popd
