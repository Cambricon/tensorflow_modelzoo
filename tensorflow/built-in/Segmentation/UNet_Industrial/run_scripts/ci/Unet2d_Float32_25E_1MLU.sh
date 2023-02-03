#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"
source env.sh
python unet2d.py \
 --exec_mode=train_and_evaluate \
 --data_dir=/data/tensorflow/training/datasets/unet2d_10type \
 --results_dir=mlu_model \
 --use_hvd=False              \
 --unet_variant=tinyUNet \
 --activation_fn=relu \
 --iter_unit=batch \
 --batch_size=16 \
 --num_iter=25 \
 --warmup_step=10           \
 --dataset_name=DAGM2007 \
 --dataset_classID=1 \
 --data_format=NHWC \
 --learning_rate=1e-4 \
 --learning_rate_decay_factor=0.8 \
 --learning_rate_decay_steps=500 \
 --rmsprop_decay=0.9             \
 --rmsprop_momentum=0.8          \
 --loss_fn_name=adaptive_loss \
 --weight_decay=1e-5 \
 --weight_init_method=he_uniform \
 --display_every=25 \
 --debug_verbosity=0 \
 --use_gpu=False \
 --use_amp=False \
 --use_performance=False \
 --use_profiler=False
popd
