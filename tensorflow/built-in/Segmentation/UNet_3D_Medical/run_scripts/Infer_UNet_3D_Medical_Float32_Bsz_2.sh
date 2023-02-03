cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

source env.sh

python unet3d.py \
    --exec_mode=evaluate \
    --data_dir=${DATA_DIR} \
    --batch_size=2\
    --model_dir=mlu_model
popd
