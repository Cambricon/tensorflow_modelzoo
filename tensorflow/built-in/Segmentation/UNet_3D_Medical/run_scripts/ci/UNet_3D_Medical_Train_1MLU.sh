set -e
cur_path=$(pwd)
work_dir="${cur_path}/../.."
pushd "${work_dir}"

source env.sh

python unet3d.py \
    --exec_mode=train \
    --data_dir=${DATA_DIR} \
    --max_steps=10 \
    --batch_size=1 \
    --use_horovod=False\
    --benchmark=False \
    --warmup_steps=200 \
    --fold=0 \
    --num_folds=5 \
    --use_performance=False \
    --use_profiler=False \
    --model_dir=mlu_model
popd
