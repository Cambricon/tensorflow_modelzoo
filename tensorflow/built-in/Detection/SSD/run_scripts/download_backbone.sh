#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

mkdir -p checkpoints
cd checkpoints
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzf resnet_v1_50_2016_08_28.tar.gz
mkdir -p resnet_v1_50
mv resnet_v1_50.ckpt resnet_v1_50/model.ckpt
cd ..

