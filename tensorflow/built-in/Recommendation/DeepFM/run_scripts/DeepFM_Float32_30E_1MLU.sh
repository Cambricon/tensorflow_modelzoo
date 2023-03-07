#!/bin/bash

dev_workspace=$(pwd)
if [[ $dev_workspace != *DeepFM* ]];
then
    echo "Please perform the training in the DeepFM workspace!"
    exit -1
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace="${dev_workspace}/.."
else
   workspace=$dev_workspace
fi
timestamp=$(date +%Y%m%d%H%M)
model_dir="${workspace}/DeepFM_model_${timestamp}"

pushd "${workspace}"

source env.sh

python deepFM.py                \
      --mode=train_and_eval     \
      --exec_mode=DeepFM        \
      --data_dir=${DATA_DIR}    \
      --model_dir=mlu_model     \
      --batch_size=1024         \
      --use_gpu=False           \
      --skip_eval=False         \
      --learning_rate=0.001     \
      --num_splits=3            \
      --epoch=30                \
      --use_horovod=False       \
      --use_amp=False           \
      --use_performance=False   \
      --use_profiler=False
popd
