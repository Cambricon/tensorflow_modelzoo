#!/bin/bash
set -e
set -x

cur_path=$(pwd)
work_dir="${cur_path}/../src"
pushd "${work_dir}"
bash prepare.sh
python main.py \
	--mode=demo \
        --use_performance=True \
	--demo_steps=60 \
	--demo_model=1521112368
popd
