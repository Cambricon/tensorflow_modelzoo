#!/bin/bash
set -e
set -x
export TF_MLU_THREAD_MODE="mlu_private"
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_MLU_THREAD_COUNT=8

# Only run 10 steps in precheckin
cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"
source env.sh
 python conformer_train.py \
 --data_dir=$DATA_DIR \
 --steps=10 \
 --batch_size=4 \
 --num_workers=1 \
 --use_gpu=False \
 --skip_eval=False \
 --use_horovod=False \
 --use_profiler=False \
 --use_performance=False \
 --mxp=False
popd
