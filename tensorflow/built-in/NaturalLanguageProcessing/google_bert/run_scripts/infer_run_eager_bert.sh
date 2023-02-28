#!/bin/bash
cur_path=$(pwd)
work_dir="${cur_path}/.."

pushd "${work_dir}"
source env.sh

input_ckpt=$1
infer_ckpt=""
if [ ! -n "$input_ckpt" ];then
    infer_ckpt="${MODEL_DIR}/bert_model.ckpt"
    echo "Warning: No input_ckpt assigned,use default ckpt to infer, which may result in a low precision result.."
else
    infer_ckpt=$input_ckpt
fi

python run_squad.py \
 --hvd_device=mlu \
 --init_checkpoint="${infer_ckpt}" \
 --max_seq_length=384 \
 --num_train_epochs=2.0 \
 --doc_stride=128 \
 --save_checkpoints_steps=1000 \
 --do_train=False \
 --do_predict=True \
 --learning_rate=1e-05 \
 --vocab_file="${MODEL_DIR}/vocab.txt" \
 --bert_config_file="${MODEL_DIR}/bert_config.json" \
 --train_file="${SQUAD_DIR}/train-v1.1.json" \
 --predict_file="${SQUAD_DIR}/dev-v1.1.json" \
 --eval_script="${SQUAD_DIR}/evaluate-v1.1.py" \
 --train_batch_size=22 \
 --output_dir=mlu_model

popd
