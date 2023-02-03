#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"
source env.sh

python3 bert_trainer.py \
  --train_steps=0 \
  --bert_config_file="${MODEL_DIR}/bert_config.json" \
  --vocab_file="${MODEL_DIR}/vocab.txt" \
  --output_dir="./bert_model_single" \
  --dllog_path="./bert_dllog.json" \
  --train_file="${SQUAD_DIR}/train-v1.1.json" \
  --predict_file="${SQUAD_DIR}/dev-v1.1.json" \
  --eval_script="${SQUAD_DIR}/evaluate-v1.1.py" \
  --init_checkpoint=""  \
  --do_lower_case=False \
  --max_seq_length=384 \
  --doc_stride=128 \
  --do_train=False \
  --do_predict=True \
  --train_batch_size=8 \
  --use_horovod=False \
  --use_profiler=False \
  --use_performance=False \
  --inter_op_threads=0 \
  --intra_op_threads=0 \
  --num_train_epochs=2 \
  --warmup_proportion=0.1 \
  --save_checkpoints_steps=1000 \
  --amp=False

popd
