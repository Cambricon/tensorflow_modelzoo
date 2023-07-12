#!/bin/bash
set -x
export MLU_VISIBLE_DEVICES=0
cur_path=$(pwd)
root_path="${cur_path}/../../"
work_dir="${root_path}/models/models/official/legacy/bert"
squad_verison=v1.1
squad_dir="${root_path}/dataset"
bert_dir="${PRETRAINED_MODEL_PATH}"
timestamp=$(date +%Y%m%d%H%M)
model_dir=${root_path}/model_output_dir_1MLU_${timestamp}
vocab_file=${bert_dir}/vocab.txt
train_data_path=${squad_dir}/squad_${squad_verison}_train.tf_record
meta_data_file_path=${squad_dir}/squad_${squad_verison}_meta_data

pushd "${work_dir}"

python run_squad.py \
  --input_meta_data_path=${meta_data_file_path} \
  --mode=train_and_eval \
  --train_data_path=${train_data_path} \
  --predict_file=${SQUAD_DATA_PATH}/dev-v1.1.json \
  --vocab_file=${bert_dir}/vocab.txt \
  --bert_config_file=${bert_dir}/bert_config.json \
  --init_checkpoint=${bert_dir}/bert_model.ckpt \
  --train_batch_size=16 \
  --predict_batch_size=4 \
  --learning_rate=8e-5 \
  --num_train_epochs=1 \
  --steps_per_epoch=50 \
  --warmup_steps=10 \
  --model_dir=${model_dir} \
  --log_steps=50 \
  --distribution_strategy=one_device
popd
