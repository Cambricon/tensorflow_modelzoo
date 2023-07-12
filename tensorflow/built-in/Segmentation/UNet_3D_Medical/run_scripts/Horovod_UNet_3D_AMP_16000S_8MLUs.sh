cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

source env.sh

horovodrun -np 8 python unet3d.py \
    --exec_mode=train \
    --use_amp=True \
    --data_dir=${DATA_DIR} \
    --max_steps=16000 \
    --batch_size=2 \
    --use_horovod=True\
    --benchmark=False \
    --warmup_steps=200 \
    --fold=0 \
    --num_folds=5 \
    --use_performance=False \
    --use_profiler=False \
    --model_dir=mlu_model
popd
