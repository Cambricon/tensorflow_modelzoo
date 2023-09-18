#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet50_model_single"

pushd "${work_dir}"

source env.sh

python3 resnet50_trainer.py \
	--mode=train_and_evaluate \
	--batch_size=128 \
	--lr_init=0.256 \
	--iter_unit=batch \
	--lr_warmup_epochs=8 \
	--warmup_steps=100 \
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
	--num_iter=10 \
	--use_horovod=False \
	--use_profiler=False \
	--use_performance=False \
	--use_amp=False

popd