#!/bin/bash
# cmd format: bash Infer_Tacotron2_Float32_1MLU.sh ./logs-Tacotron-2/taco_pretrained/
dev_workspace=$(pwd)
if [[ $dev_workspace != *Tacotron* ]];
then
    echo "Please perform the training in the tacotron2 workspace!"
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

pushd "${workspace}"

source env.sh
bash install_dependency.sh

python synthesize.py              \
       --tacotron_num_devices=1   \
       --tacotron_batch_size=32   \
       --tacotron_synthesis_batch_size=1   \
       --device_type=mlu                   \
       --taco_checkpoint=${CKPT_TACOTRON}  \
       --mels_dir=tacotron_output/eval/

popd
