#!/bin/bash
dev_workspace=$(pwd)
if [[ $dev_workspace != *Tacotron* ]];
then
    echo "Please perform the training in the Tacotron2 workspace!"
    exit -1
elif [[ $dev_workspace == *ci* ]]
then
   workspace="${dev_workspace}/../.."
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace="${dev_workspace}/.."
else
   workspace=$dev_workspace
fi
timestamp=$(date +%Y%m%d%H%M)
model_dir="${workspace}/Tacotron2_model_${timestamp}"

pushd "${workspace}"

source env.sh
bash install_dependency.sh

horovodrun -np 4 python train.py       \
         --tacotron_num_devices=1      \
         --tacotron_batch_size=32      \
         --tacotron_synthesis_batch_size=1  \
         --device_type=mlu             \
         --checkpoint_interval=2000    \
         --tacotron_train_steps=20     \
         --use_amp=False               \
         --input_dir=${DATA_DIR}            \
         --tacotron_input=${DATA_DIR}/train.txt  \
         --output_dir=mlu_model        \
         --use_horovod=True            \
         --use_profiler=False          \
         --use_performance=False

rm -rf logs-Tacotron-2

popd
