source env.sh
horovodrun -np 8 python -u dlrm.py \
         --dataset_path ${DATA_DIR} \
         --interaction=dot_tensorflow \
         --mlu \
         --xla --amp \
         --inter_op_parallelism=16 \
         --intra_op_parallelism=16
