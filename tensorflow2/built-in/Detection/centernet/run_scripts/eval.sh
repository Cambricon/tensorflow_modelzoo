cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
python centernet_train.py\
        --pipeline_config_path=./models/mlu_centernet_config.config\
       	--model_dir=mlu_model\
       	--checkpoint_dir=./mlu_model\
       	--load_latest_ckpt_continuously=False\
       	--do_train=False
popd
