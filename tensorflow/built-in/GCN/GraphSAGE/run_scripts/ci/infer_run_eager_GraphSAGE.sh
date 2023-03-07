#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"

source env.sh

python3 graphsage_infer.py \
  --data_dir=${DATA_DIR} \
  --embed_dir="./GraphSAGE_model_float32/unsup-ppi/graphsage_mean_small_0.000010" \
  --setting=test

popd
