#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"
source env.sh
horovodrun -np 4 python transformer.py \
	--batch_size=4096 \
	--max_length=64 \
	--static_batch=True \
	--num_gpus=0 \
	--num_mlus=1 \
	--distribution_strategy=one_device \
	--enable_time_history=True \
	--use_synthetic_data=False \
	--enable_checkpointing=True \
	--save_ckpt_steps=1000 \
	--enable_tensorboard=False \
	--mode=train \
	--inter_op_threads=0 \
        --intra_op_threads=0 \
	--data_dir=$DATA_DIR \
	--model_dir=mlu_model \
	--vocab_file=$DATA_DIR/vocab.ende.32768 \
	--param_set=base \
	--train_steps=80 \
	--steps_between_evals=8 \
	--use_horovod=True \
	--use_profiler=False \
	--use_performance=False
popd