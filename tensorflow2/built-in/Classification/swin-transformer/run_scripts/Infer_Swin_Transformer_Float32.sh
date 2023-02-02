#!/bin/bash
# cmd format: bash Infer_Swin_Transformer_Float32_1MLU.sh ./swin-transformer_model/model.ckpt-0012
dev_workspace=$(pwd)
if [[ $dev_workspace != *swin* ]];
then
    echo "Please perform the training in the swin-transformer workspace!"
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
    bash Infer_Swin_Transformer_Float32_1MLU.sh ./swin-transformer_model/model.ckpt-0012 \033[0m"
   exit -1
fi

pushd "${workspace}"

source env.sh

python swin_infer.py                    \
	--dataset=imagenet              \
	--mode=eval                     \
	--data_dir=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord  \
	--checkpoint_file=${checkpoint} \
	--model_name=swin_large_224     \
	--batch_size=16                 \
	--one_hot=False
popd
