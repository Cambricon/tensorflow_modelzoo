#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."
timestamp=$(date +%Y%m%d%H%M)
model_dir="${work_dir}/bert_model_hvd_amp_${timestamp}"

pushd "${work_dir}"

source env.sh
horovodrun -np 8 python run_squad.py \
 --use_horovod=True \
 --use_amp=True \
 --finetune_steps=0 \
 --hvd_device=mlu \
 --max_seq_length=384 \
 --num_train_epochs=2.0 \
 --doc_stride=128 \
 --save_checkpoints_steps=1000 \
 --do_train=True \
 --do_predict=True \
 --learning_rate=1e-5 \
 --vocab_file="${MODEL_DIR}/vocab.txt" \
 --bert_config_file="${MODEL_DIR}/bert_config.json" \
 --init_checkpoint="${MODEL_DIR}/bert_model.ckpt" \
 --train_file="${SQUAD_DIR}/train-v1.1.json" \
 --predict_file="${SQUAD_DIR}/dev-v1.1.json" \
 --eval_script="${SQUAD_DIR}/evaluate-v1.1.py" \
 --train_batch_size=22 \
 --output_dir=$model_dir \
 --use_profiler=False \
 --use_performance=False

popd
