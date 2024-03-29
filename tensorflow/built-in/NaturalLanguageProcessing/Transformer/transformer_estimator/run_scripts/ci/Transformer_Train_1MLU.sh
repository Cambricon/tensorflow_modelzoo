#!/bin/bash
set -e
set -x
cur_path=$(pwd)
work_dir="${cur_path}/../.."
model_dir="${work_dir}/mlu_model"
if [ -d ${model_dir} ];
then
  rm -r "${model_dir}"/*
fi

export MLU_VISIBLE_DEVICES=0
pushd "${work_dir}"
# fp32 1mlu
source env.sh

bleu_src="${DATA_DIR}/newstest2014.en"
bleu_ref="${DATA_DIR}/newstest2014.de"
vocab_file="${DATA_DIR}/vocab.ende.32768"
 python transformer_main.py \
 --mode=train \
 --batch_size=4096 \
 --train_steps=10 \
 --steps_between_evals=15 \
 --bleu_source=$bleu_src \
 --bleu_ref=$bleu_ref \
 --data_dir=$DATA_DIR \
 --vocab_file=$vocab_file \
 --model_dir=$model_dir \
 --num_gpus=0 \
 --num_mlus=1 \
 --param_set=base \
 --use_amp=False \
 --use_profiler=False \
 --use_horovod=False \
 --use_performance=False 
popd

