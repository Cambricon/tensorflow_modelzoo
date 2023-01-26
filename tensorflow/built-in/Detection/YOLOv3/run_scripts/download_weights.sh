#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."

pushd "${work_dir}"

if [ ! -d "checkpoint" ]; then
  mkdir checkpoint
fi
cd checkpoint
if [ ! -f "yolov3_coco.tar.gz" ]; then
  wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
fi

tar -xvf yolov3_coco.tar.gz
cd ..
export PYTHONPATH=${work_dir}
python ${work_dir}/models/convert_weight.py --train_from_coco
