#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/dlrm_model_${timestamp}"
set -e

pushd "${work_dir}"

source env.sh
mkdir $model_dir

horovodrun -np 4 python3 main.py \
    --dataset_type=raw \
    --dataset_path=$DATA_DIR  \
    --save_checkpoint_path=$model_dir/ \
    --batch_size=65536 \
    --valid_batch_size=65536 \
    --dummy_model=False \
    --dummy_embedding=False \
    --learning_rate=24 \
    --optimizer=sgd \
    --num_numerical_features=13 \
    --synthetic_dataset_train_batches=64008 \
    --synthetic_dataset_valid_batches=1350 \
    --use_amp=True \
    --xla=False \
    --loss_scale=1024 \
    --prefetch_batches=10 \
    --auc_thresholds=8000 \
    --epochs=1 \
    --run_eagerly=False \
    --embedding_type=split_embedding \
    --embedding_trainable=True \
    --dot_interaction=tensorflow \
    --embedding_dim=128 \
    --evals_per_epoch=1 \
    --print_freq=1 \
    --warmup_steps=8000 \
    --decay_start_step=48000 \
    --decay_steps=24000 \
    --profiler_start_step=1 \
    --profiler_steps=1 \
    --profiled_rank=0 \
    --inter_op_parallelism=0 \
    --intra_op_parallelism=0 \
    --tf_mlu_memory_limit_gb=24 \
    --data_parallel_bottom_mlp=False \
    --experimental_columnwise_split=False \
    --log_path=dlrm_tf_log.json \
    --use_mlus=True \
    --use_gpus=False \
    --skip_eval=False \
    --benchmark_warmup_steps=0 \
    --max_steps=10 \
    --use_horovod=True \
    --use_profiler=False \
    --use_performance=False
popd
