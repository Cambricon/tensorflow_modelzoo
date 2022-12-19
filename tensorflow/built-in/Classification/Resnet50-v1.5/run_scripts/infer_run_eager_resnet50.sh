#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
source env.sh

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --model_dir)
      model_dir="${work_dir}/${2}"
      shift 2
      ;;
    *)
      echo "Unknown argument ${key}. Please use 'bash infer_run_eager_resnet50.sh --model_dir network_model_dir' to train the network"
      exit 1
      ;;
  esac
done

python3 resnet50_trainer.py \
	--mode=evaluate \
	--batch_size=128 \
	--lr_init=0.256 \
	--iter_unit=epoch \
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
	--num_iter=1 \
	--use_horovod=False \
	--use_profiler=False \
	--use_performance=False \
	--use_amp=False

popd