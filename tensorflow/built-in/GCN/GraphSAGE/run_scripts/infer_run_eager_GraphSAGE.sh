#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

source env.sh

if [ ! -n "$1" ];then
  echo "Please use 'bash infer_run_eager_GraphSAGE.sh PATH_TO_CKPT' to infer."
  exit 1
else
  PATH_TO_CKPT=${1}
fi

python3 graphsage_infer.py \
  --data_dir=${DATA_DIR} \
  --embed_dir=${PATH_TO_CKPT} \
  --setting=test

popd