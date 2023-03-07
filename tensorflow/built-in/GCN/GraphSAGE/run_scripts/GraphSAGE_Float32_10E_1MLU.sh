#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/GraphSAGE_model_${timestamp}"
  
pushd "${work_dir}"

source env.sh

python3 bert_trainer.py \
  --base_log_dir=${model_dir} \
  --data_dir="${DATA_DIR}/ppi" \
  --epochs=10 \
  --batch_size=512 \
  --use_amp=False \
  --use_profiler=False \
  --use_performance=False

popd
