#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
source env.sh
python BERT_NER.py \
	--task_name=NER \
	--hvd_device=mlu \
	--use_amp=False \
	--do_lower_case=False \
	--max_seq_length=128 \
	--doc_stride=128 \
	--do_train=False \
	--do_eval=False \
	--do_predict=True \
	--learning_rate=2e-05 \
	--predict_batch_size=8 \
	-vocab_file=$DATA_DIR/cased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=$DATA_DIR/cased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=$DATA_DIR/cased_L-12_H-768_A-12/bert_model.ckpt \
	--data_dir=$DATA_DIR/data \
	--output_dir=mlu_model
popd
