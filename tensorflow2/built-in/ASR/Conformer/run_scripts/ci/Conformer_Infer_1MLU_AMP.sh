#!/bin/bash
export TF_MLU_THREAD_MODE="mlu_private"
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_MLU_THREAD_COUNT=8
export MLU_VISIBLE_DEVICES=0

cur_path=$(pwd)
work_dir="${cur_path}/../.."
bsz=32
output_file="${work_dir}/test_${bsz}_amp.tsv"

pushd "${work_dir}"
source env.sh
python conformer_test.py \
 --data_dir=$DATA_DIR \
 --batch_size=$bsz \
 --output=$output_file  \
 --get_rtf=True  \
 --mxp=True  \
 --saved=$INFER_CKPT
popd
