#!/bin/bash
set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
model_dir="${work_dir}/GraphSAGE_model_float32"

pushd "${work_dir}"

source env.sh

python3 graphsage_trainer.py \
  --base_log_dir="${model_dir}" \
  --data_dir="${DATA_DIR}/ppi" \
  --batch_size=512 \
  --use_amp=False \
  --max_total_steps=10 \
  --use_profiler=False \
  --use_performance=False

popd
