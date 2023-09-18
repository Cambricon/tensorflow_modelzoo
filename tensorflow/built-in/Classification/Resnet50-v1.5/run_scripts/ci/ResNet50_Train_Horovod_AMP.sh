#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet50_model_amp"

pushd "${work_dir}"

source env.sh

horovodrun -np 8 python3 resnet50_trainer.py \
	--mode=train_and_evaluate \
	--batch_size=768 \
	--lr_init=0.768 \
	--iter_unit=batch \
	--lr_warmup_epochs=24 \
	--warmup_steps=300 \
	--weight_init=fan_in \
	--export_dir="./export_dir" \
	--run_iter=-1 \
	--label_smoothing=0.1 \
	--momentum=0.875 \
	--weight_decay=3.0517578125e-05 \
	--data_format=NHWC \
	--results_dir="."  \
	--use_dali=False \
	--model_dir=$model_dir \
	--finetune_checkpoint="" \
	--data_dir=$DATA_DIR \
	--data_idx_dir=$DATA_IDX_DIR \
	--num_iter=40 \
	--use_horovod=True \
	--use_profiler=False \
	--use_performance=False \
	--use_amp=True

popd
