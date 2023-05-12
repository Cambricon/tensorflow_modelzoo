#!/bin/bash
set -x

cur_path=$(pwd)
root_path="${cur_path}/.."
models_path=${root_path}/models/models
max_seq_length=384
squad_verison=v1.1
squad_dir=${SQUAD_DATA_PATH}
bert_dir=${PRETRAINED_MODEL_PATH}
squad_data_file=${squad_dir}/train-${squad_verison}.json
vocab_file=${bert_dir}/vocab.txt

train_data_output_path=${squad_dir}/squad_${squad_verison}_train.tf_record
meta_data_file_path=${squad_dir}/squad_${squad_verison}_meta_data

pushd ${models_path}
    if [ ! -f "${train_data_output_path}" ];then
        echo "cambricon-note: Generating ${train_data_output_path} and ${meta_data_file_path} ......"
        python official/nlp/data/create_finetuning_data.py --squad_data_file=${squad_data_file} \
         --vocab_file=${vocab_file} \
         --train_data_output_path=${train_data_output_path} \
         --meta_data_file_path=${meta_data_file_path} \
         --fine_tuning_task_type=squad \
         --max_seq_length=${max_seq_length}
    else
        echo "cambricon-note: ${train_data_output_path} and ${meta_data_file_path} have already been generated."
    fi
popd
