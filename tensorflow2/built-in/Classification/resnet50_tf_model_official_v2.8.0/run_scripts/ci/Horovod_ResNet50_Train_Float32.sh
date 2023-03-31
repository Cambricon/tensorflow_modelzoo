#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/resnet50_model_${timestamp}"

pushd "${work_dir}"
set -e
source env.sh

horovodrun -np 4 python3 resnet_main.py \
    --model_dir=${model_dir} \
    --data_dir=${DATA_DIR} \
    --mode=train \
    --num_mlus=1 \
    --num_gpus=0 \
    --distribution_strategy=off \
    --batch_size=128 \
    --steps_per_loop=1 \
    --train_epochs=90 \
    --use_synthetic_data=False \
    --use_performance=False \
    --use_amp=False \
    --use_horovod=True \
    --log_steps=1 \
    --run_eagerly=False \
    --enable_checkpoint_and_export=True \
    --base_learning_rate=0.1 \
    --train_steps=10 \
    --use_profiler=False \
    --enable_tensorboard=False \
    --dtype=fp32 \
    --tf_mlu_thread_mode=mlu_private \
    --tf_gpu_thread_mode=gpu_private \
    --per_mlu_thread_count=2 \
    --per_gpu_thread_count=2 \
    --inter_op_parallelism_threads=4 \
    --intra_op_parallelism_threads=8 \
    --datasets_num_private_threads=2 \
    --epochs_between_evals=4 \
    --host_tracer_level=2 \
    --device_tracer_level=1 \
    --profiler_dir=${model_dir}
popd
