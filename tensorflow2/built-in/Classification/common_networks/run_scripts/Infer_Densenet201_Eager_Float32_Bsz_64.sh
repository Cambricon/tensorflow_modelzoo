#!/bin/bash
set -x
cur_path=$(pwd)
work_dir="${cur_path}/.."
batch_size=64
use_amp=False
pushd "${work_dir}"
source env.sh
popd
bash run_infer_networks.sh --model_type "densenet201" \
 --infer_ckpt "${DENSENET201_CKPT}" \
 --batch_size ${batch_size} \
 --use_amp ${use_amp}
