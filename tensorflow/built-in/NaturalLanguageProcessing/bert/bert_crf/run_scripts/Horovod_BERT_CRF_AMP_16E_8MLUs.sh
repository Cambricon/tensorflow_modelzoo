#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
rm -rf mlu_model
source env.sh
horovodrun -np 8 python BERT_NER.py \
	--task_name=NER \
	--do_lower_case=False \
	--crf=True \
	--use_horovod=True \
	--use_amp=True \
	--finetune_steps=0 \
	--hvd_device=mlu \
	--max_seq_length=128 \
	--num_train_epochs=16.0 \
	--save_checkpoints_steps=1000 \
	--do_train=True \
	--do_eval=True \
	--do_predict=True \
	--learning_rate=1.05e-5 \
	--vocab_file=$DATA_DIR/cased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=$DATA_DIR/cased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=$DATA_DIR/cased_L-12_H-768_A-12/bert_model.ckpt \
	--data_dir=$DATA_DIR/data \
	--train_batch_size=32 \
	--output_dir=mlu_model \
	--use_profiler=False \
	--use_performance=False
popd
