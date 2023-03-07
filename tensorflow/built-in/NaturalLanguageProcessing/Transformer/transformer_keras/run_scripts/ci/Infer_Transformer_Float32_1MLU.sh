#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"
source env.sh
python transformer.py \
	--batch_size=4096 \
	--max_length=64 \
	--static_batch=True \
	--num_gpus=0 \
	--num_mlus=1 \
	--distribution_strategy=one_device \
	--enable_time_history=False \
	--use_synthetic_data=False \
	--enable_tensorboard=False \
	--mode=eval \
	--bleu_source=$DATA_DIR/newstest2014.en \
	--bleu_ref=$DATA_DIR/newstest2014.de \
	--data_dir=$DATA_DIR \
	--model_dir=mlu_model \
	--vocab_file=$DATA_DIR/vocab.ende.32768 \
	--param_set=base \
	--use_profiler=False \
	--use_performance=False
popd
