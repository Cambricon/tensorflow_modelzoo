#!/bin/bash
set -x

cur_path=$(pwd)
models_path="${cur_path}/../models"
patch_file="${models_path}/modification_for_mlu.patch"

# 1. get pretrained_model()
if [ -z "${PRETRAINED_MODEL_PATH}" ];then
    echo "Error: PRETRAINED_MODEL_PATH is not set , please set it in env.sh and export it to ENV!"
    exit 1
else
    if [ ! -d "${PRETRAINED_MODEL_PATH}" ];then
        mkdir -p "${PRETRAINED_MODEL_PATH}"
        pushd "${PRETRAINED_MODEL_PATH}"
            wget -c https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz
            mkdir tmp_dir
            tar -xvf uncased_L-12_H-768_A-12.tar.gz -C "${PRETRAINED_MODEL_PATH}/tmp_dir"
            mv ${PRETRAINED_MODEL_PATH}/tmp_dir/uncased_L-12_H-768_A-12/* .
            rm -rf tmp_dir

        popd
    fi
fi

# 2. git clone original model and git apply patch
pushd "${models_path}"
dst_dir="${models_path}/models"
if [ ! -d "${dst_dir}" ];then
    git clone https://github.com/tensorflow/models.git -b v2.9.0
    cp "${patch_file}" "${dst_dir}"
    cd "${dst_dir}"
    git apply "${patch_file}"
    cd -
fi
popd

# 3. install requirements
pushd "${cur_path}/.."
apt update
apt install -y protobuf-compiler
pip install -r requirements.txt
popd

# 4. get ori data(squad)
squad_data_url_file=squad_data_url
if [ ! -d "${SQUAD_DATA_PATH}" ];then
    mkdir -p "${SQUAD_DATA_PATH}"
    wget -i "${squad_data_url_file}" -P "${SQUAD_DATA_PATH}"

fi
# 5. generate train data(tf_record)
bash generate_train_data.sh

