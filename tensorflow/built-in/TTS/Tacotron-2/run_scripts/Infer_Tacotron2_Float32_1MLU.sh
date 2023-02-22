#!/bin/bash
# cmd format: bash Infer_Tacotron2_Float32_1MLU.sh ./logs-Tacotron-2/taco_pretrained/
dev_workspace=$(pwd)
if [[ $dev_workspace != *Tacotron* ]];
then
    echo "Please perform the training in the tacotron2 workspace!"
    exit -1
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace="${dev_workspace}/.."
else
   workspace=$dev_workspace
fi

checkpoint=$1
if [[ ! -n $checkpoint ]];
then
   echo -e "\033[31m Checkpoint is not passed in, please check the command! \033[0m"
   echo -e "\033[34m Execute Command Reference : \
    bash Infer_Tacotron2_Float32_1MLU.sh ./logs-Tacotron-2/taco_pretrained/ \033[0m"
   exit -1
fi

pushd "${workspace}"

source env.sh

python synthesize.py              \
       --tacotron_num_devices=1   \
       --tacotron_batch_size=32   \
       --tacotron_synthesis_batch_size=1   \
       --device_type=mlu                   \
       --taco_checkpoint=$checkpoint \
       --mels_dir=tacotron_output/eval/

popd
