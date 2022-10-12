#!/bin/bash
export TF_MLU_THREAD_MODE="mlu_private"
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_MLU_THREAD_COUNT=8

cur_path=$(pwd)
work_dir="${cur_path}/.."
source env.sh
ckpt_dir="${work_dir}/mlu_model/checkpoints/50.h5"
bsz=1
output_file="${work_dir}/test_${bsz}.tsv"
pushd "${work_dir}"
python conformer_test.py --data_dir=$DATA_DIR --batch_size=$bsz --output=$output_file  --saved=$ckpt_dir 
popd
