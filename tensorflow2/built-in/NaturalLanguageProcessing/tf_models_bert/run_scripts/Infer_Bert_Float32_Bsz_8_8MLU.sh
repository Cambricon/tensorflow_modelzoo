#!/bin/bash
set -x
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cur_path=$(pwd)
root_path="${cur_path}/.."
model_dir=${1}
# set model_dir to the checkpoint path before run this script.
# for instance:
#model_dir=${root_path}/model_output_dir_8MLUs_202306010803

work_dir="${root_path}/models/models/official/legacy/bert"
squad_verison=v1.1
squad_dir="${root_path}/dataset"
bert_dir="${PRETRAINED_MODEL_PATH}"
timestamp=$(date +%Y%m%d%H%M)
vocab_file=${bert_dir}/vocab.txt
train_data_path=${squad_dir}/squad_${squad_verison}_train.tf_record
meta_data_file_path=${squad_dir}/squad_${squad_verison}_meta_data

if [ ! -d "${model_dir}" ] || [ ! -f ${model_dir}/checkpoint ];then
    echo "${model_dir} or ${model_dir}/checkpoint is not found. Please check!"
    exit 1
fi

pushd "${work_dir}"

python run_squad.py \
  --input_meta_data_path=${meta_data_file_path} \
  --mode=eval \
  --train_data_path=${train_data_path} \
  --predict_file=${SQUAD_DATA_PATH}/dev-v1.1.json \
  --vocab_file=${bert_dir}/vocab.txt \
  --bert_config_file=${bert_dir}/bert_config.json \
  --init_checkpoint=${bert_dir}/bert_model.ckpt \
  --predict_batch_size=8 \
  --model_dir=${model_dir} \
  --num_gpus=8 \
  --distribution_strategy=mirrored
popd
