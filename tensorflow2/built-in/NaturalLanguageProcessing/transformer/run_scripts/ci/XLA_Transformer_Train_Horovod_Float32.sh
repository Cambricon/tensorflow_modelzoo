#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../../"
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/mlu_model_${timestamp}"

pushd "${work_dir}"
# fp32 8mlu
source env.sh

bleu_src="${DATA_DIR}/newstest2014.en"
bleu_ref="${DATA_DIR}/newstest2014.de"
vocab_file="${DATA_DIR}/vocab.ende.32768"
horovodrun -np 8 python transformer_main.py \
 --mode=train \
 --batch_size=4096 \
 --max_length=64 \
 --train_steps=10 \
 --steps_between_evals=100000 \
 --bleu_source=$bleu_src \
 --bleu_ref=$bleu_ref \
 --data_dir=$DATA_DIR \
 --vocab_file=$vocab_file \
 --model_dir=$model_dir \
 --num_gpus=0 \
 --num_mlus=1 \
 --distribution_strategy=off \
 --enable_checkpointing=True \
 --param_set=base \
 --enable_time_history=False \
 --tf_mlu_thread_mode=mlu_private \
 --tf_gpu_thread_mode=gpu_private \
 --per_mlu_thread_count=4 \
 --per_gpu_thread_count=4 \
 --inter_op_parallelism_threads=4 \
 --intra_op_parallelism_threads=4 \
 --horovod_fusion_threshold=33554432 \
 --use_amp=False \
 --use_profiler=False \
 --use_horovod=True \
 --use_performance=False \
 --enable_xla=True

popd
