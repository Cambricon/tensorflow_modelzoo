#!/bin/bash
set -e
set -x

export MLU_VISIBLE_DEVICES=0

cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)

models_dir="${work_dir}/models"
features=$work_dir/test_features.f32
output_dir=mlu_model_infer_${timestamp}
tmp_output=lpcnet_infer_output.s16
wav_output=lpcnet_infer_output_${timestamp}.wav


pushd $work_dir
# generate test features based on .wav
source env.sh
test_wav_file="${DATA_DIR}/MA/MA04_10.wav"
$models_dir/dump_data -test $test_wav_file $features

# infer
python lpcnet_test.py \
 --checkpoint $INFER_CKPT \
 --features $features \
 --output_dir $output_dir \
 --output $tmp_output

# convert infer result to wav file
ffmpeg -f s16le -ar 16k -ac 1 -i "${output_dir}/${tmp_output}" $wav_output 

popd
