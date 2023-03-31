#!/bin/bash
set -x
cur_path=$(pwd)
work_dir="${cur_path}/.."
batch_size=128
use_amp=False
pushd "${work_dir}"
source env.sh
popd
bash run_infer_networks.sh --model_type "resnet18" \
 --infer_ckpt "${RESNET18_CKPT}" \
 --batch_size ${batch_size} \
 --use_amp ${use_amp}
