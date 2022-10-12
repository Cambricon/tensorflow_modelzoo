#!/bin/bash
set -e
set -x

#--------------------------------------------------------
# Before run this script,
# Modify the below UPPER var according to your own path.
# Follow the README.md to generate $features
#--------------------------------------------------------

cur_path=$(pwd)
work_dir="${cur_path}/../"
models_dir="${work_dir}/models"

pushd $work_dir
# modify the ckpt path and name according to your own environment
ckpt=YOUR_CKPT_PATH/lpcnet_model_384_99.h5
features=YOUR_INFER_DATA_PATH/test_features.f32
output_dir=mlu_model_infer
tmp_output=lpcnet_infer_output.s16
wav_output=lpcnet_infer_output.wav

# infer
python lpcnet_test.py --checkpoint $ckpt \
 --features $features \
 --output_dir $output_dir \
 --output $tmp_output

# convert infer result to wav file
ffmpeg -f s16le -ar 16k -ac 1 -i "${output_dir}/${tmp_output}" $wav_output 

popd
