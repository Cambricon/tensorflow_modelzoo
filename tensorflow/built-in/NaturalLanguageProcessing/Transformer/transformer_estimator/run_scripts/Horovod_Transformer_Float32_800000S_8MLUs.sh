#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../"
model_dir="${work_dir}/mlu_model"

export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
pushd "${work_dir}"
# fp32 8mlu
source env.sh

bleu_src="${DATA_DIR}/newstest2014.en"
bleu_ref="${DATA_DIR}/newstest2014.de"
vocab_file="${DATA_DIR}/vocab.ende.32768"
horovodrun -np 8 python transformer_main.py \
 --mode=train \
 --train_steps=100000 \
 --steps_between_evals=100000 \
 --bleu_source=$bleu_src \
 --bleu_ref=$bleu_ref \
 --batch_size=4096 \
 --data_dir=$DATA_DIR \
 --vocab_file=$VOCAB_FILE \
 --model_dir=$model_dir \
 --num_gpus=0 \
 --num_mlus=1 \
 --param_set=base \
 --use_amp=False \
 --use_profiler=False \
 --use_horovod=True \
 --use_performance=False 
popd
