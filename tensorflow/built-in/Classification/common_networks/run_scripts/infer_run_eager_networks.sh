#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
source env.sh

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --model_type)
      model_type=${2}
      shift 2
      ;;
    --model_dir)
      model_dir=${2}
      shift 2
      ;;
    *)
      echo "Unknow argument ${key}. Please use `bash infer_run_eager_networks.sh --model_type network --model_dir=network_model_dir` to train the network"
      exit 1
      ;;
  esac
done

python3 classifier_trainer.py \
    --dataset=imagenet \
    --model_type=$model_type \
    --mode=eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=0 \
    --epochs=0 \
    --start_epoch=0 \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=False \
    --skip_eval=False \
    --finetune_steps=0 \
    --finetune_checkpoint="" \
    --validation_steps=0 \
    --use_qat=False \
    --inter_op_threads=0 \
    --intra_op_threads=0 \
    --use_dummy_synthetic_data=False
popd
