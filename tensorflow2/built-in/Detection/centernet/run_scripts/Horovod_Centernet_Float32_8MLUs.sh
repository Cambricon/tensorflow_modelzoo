cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"
horovodrun -np 8 python centernet_train.py\
    --mode=train\
    --num_workers=1\
    --checkpoint_every_n=1000\
    --eval_on_train_data=False\
    --pipeline_config_path=./models/mlu_centernet_config.config\
    --batch_size=8\
    --num_train_steps=140000\
    --num_steps_per_iter=100\
    --model_dir=mlu_model\
    --do_train=True\
    --use_gpus=False\
    --use_profiler=False\
    --use_performance=False\
    --use_horovod=True\
    --use_amp=False
popd
