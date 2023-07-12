#!/bin/bash
cur_path=$(pwd)
root_path=${cur_path}/../../

pushd ${root_path}
    source env.sh
popd

pushd ${root_path}/run_scripts
    bash prepare.sh
popd
