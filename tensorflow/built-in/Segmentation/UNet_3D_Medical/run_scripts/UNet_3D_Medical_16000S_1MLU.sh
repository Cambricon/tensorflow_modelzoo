cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

source env.sh

python unet3d.py \
    --exec_mode=train \
    --data_dir=${DATA_DIR} \
    --max_steps=16000 \
    --model_dir=mlu_model \
    --use_performance=False \
    --use_profiler=False \
    --augment=True \
    --batch_size=2 \
    --use_amp=False \
    --benchmark=False \
    --warmup_steps=200 \
    --fold=0 \
    --num_folds=5 \
    --resume_training=False
popd
