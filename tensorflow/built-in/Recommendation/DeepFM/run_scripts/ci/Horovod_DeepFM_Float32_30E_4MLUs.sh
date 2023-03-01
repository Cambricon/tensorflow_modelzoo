#!/bin/bash
dev_workspace=$(pwd)
if [[ $dev_workspace != *DeepFM* ]];
then
    echo "Please perform the training in the DeepFM workspace!"
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
model_dir="${workspace}/DeepFM_model_${timestamp}"

pushd "${workspace}"

source env.sh

horovodrun -np 4 python deepFM.py   \
	--mode=train_and_eval       \
	--exec_mode=DeepFM          \
	--data_dir=${DATA_DIR}      \
       	--model_dir=mlu_model       \
	--batch_size=1024           \
	--use_gpu=False             \
	--skip_eval=True            \
	--learning_rate=0.001       \
	--num_splits=3              \
	--finetune_steps=40         \
	--epoch=1                   \
	--use_horovod=True          \
	--use_amp=False             \
	--use_performance=False     \
	--use_profiler=False

popd

