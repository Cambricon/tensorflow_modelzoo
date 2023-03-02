#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
precision=""

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --model_type)
      model_type=${2}
      shift 2
      ;;
    --infer_ckpt)
      infer_ckpt=${2}
      shift 2
      ;;
    --batch_size)
      batch_size=${2}
      shift 2
      ;;
    --use_amp)
      use_amp=${2}
      shift 2
      ;;
    *)
      echo "Unknow argument ${key}. Please use `bash run_infer_networks.sh --model_type network --infer_ckpt infer_ckpt --batch_size batch_size --use_amp use_amp`to test the network"
      echo "For instance:bash run_infer_networks.sh --model_type resnet50 --infer_ckpt path/to/model.ckpt-0001 --batch_size 128 --use_amp True"
      exit 1
      ;;
  esac
done

pushd "${work_dir}"
#source env.sh

if [ $use_amp = True ];then
    precision="amp"
else
    precision="fp32"
fi

model_dir="${work_dir}/${model_type}_model_${precision}_${timestamp}"
python3 classifier.py \
    --dataset=imagenet \
    --model_type=$model_type \
    --mode=eval \
    --model_dir=$model_dir \
    --data_dir=$DATA_DIR \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=${batch_size} \
    --epochs=90 \
    --use_performance=False \
    --use_amp=${use_amp} \
    --use_horovod=False \
    --run_eagerly=False \
    --skip_eval=False \
    --finetune_steps=0 \
    --finetune_checkpoint="${infer_ckpt}" \
    --enable_tensorboard=False \
    --datasets_num_private_threads=0
popd

