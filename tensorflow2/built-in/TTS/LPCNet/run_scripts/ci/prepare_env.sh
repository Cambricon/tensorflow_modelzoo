#!/bin/bash
set -e
set -x

cur_path=$(pwd)
models_dir="${cur_path}/../../models"

pushd $models_dir
# compile src
    apt-get update -y
    apt-get install -y autoconf automake libtool sox ffmpeg 
    ./autogen.sh
    ./configure
    make
popd
