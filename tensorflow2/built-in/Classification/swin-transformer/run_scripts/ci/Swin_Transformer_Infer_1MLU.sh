#!/bin/bash
# Only run 10 steps in precheck
set -e
set -x
dev_workspace=$(pwd)
if [[ $dev_workspace != *swin* ]];
then
    echo "Please perform the training in the swin-transformer workspace!"
    exit -1
elif [[ $dev_workspace == *ci* ]]
then
   workspace=${dev_workspace}/../..
elif [[ $dev_workspace == *run_scripts* ]]
then
   workspace=${dev_workspace}/..
else
   workspace=${dev_workspace}
fi

timestamp=$(date +%Y%m%d%H%M)
model_dir="${workspace}/swin-transformer_model_${timestamp}"

pushd "${workspace}"

source env.sh

python swin_infer.py                    \
	--dataset=imagenet              \
	--mode=eval                     \
	--data_dir=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord  \
	--model_name=swin_large_224     \
	--batch_size=16                 \
	--one_hot=False

popd
