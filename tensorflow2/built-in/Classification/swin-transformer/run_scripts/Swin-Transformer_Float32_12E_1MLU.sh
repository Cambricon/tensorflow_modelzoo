#!/bin/bash
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
timestamp=$(date +%Y%m%d%H%M)
model_dir="${workspace}/swin-transformer_model_${timestamp}"

pushd "${workspace}"

source env.sh

python swin_trainer.py          \
      --dataset=imagenet        \
      --mode=train_and_eval     \
      --data_dir=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord \
      --model_dir=${model_dir}     \
      --model_name=swin_large_224 \
      --batch_size=24             \
      --finetune_steps=0          \
      --epochs=12                 \
      --one_hot=False             \
      --skip_eval=True            \
      --validation_steps=0        \
      --use_horovod=False         \
      --use_profiler=False  \
      --use_performance=True      \
      --use_dummy_synthetic_data=False \
      --num_mlus=1                     \
      --num_gpus=0                     \
      --use_amp=True                   \
      --distribution_strategy=mirrored \
      --enable_xla=False
popd
