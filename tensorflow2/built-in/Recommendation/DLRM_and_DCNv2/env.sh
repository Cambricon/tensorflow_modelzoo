pip_list=`pip list`

regex="cnhorovod-tensorflow *0.14.7"
if [[ "${pip_list}" =~ $regex ]];then
  echo "cnhorovod-tensorflow 0.14.7 has been installed"
else
  pip install wget http://daily.software.cambricon.com/release/horovod/0.14.7/src/horovod-v0.14.7.tar.gz
fi

regex="embedding-table-plugin *0.1.0"
if [[ "${pip_list}" =~ $regex ]];then
  echo "embedding-table-plugin 0.1.0 has been installed"
else
  pip install wget http://daily.software.cambricon.com/daily/embedding-table-plugin/src/0.1.0/embedding_table_plugin-0.1.0.tar.gz
fi

pip install -r mlu_requirements.txt


export TF_MLU_ALLOW_GROWTH=true
export TF_MLU_THREAD_MODE="mlu_private"
export TF_MLU_THREAD_COUNT=4

export TF_MLU_UNIQUE_PRESERVE_ORDER=false

export TF_MLU_SYNC_D2H=false
export TF_MLU_SYNC_H2D=false

export TF_MLU_TS_ALLOCATOR=true
export TF_MLU_KT_MAX_INTERVAL=128
export TF_MLU_KT_MAX_BYTES=0
export TF_MLU_KT_MAX_PENDING=0

export HOROVOD_CYCLE_TIME=0.001
export HOROVOD_ENABLE_XLA_OPS=true

export TF_XLA_DO_NOT_COMPILE_POSSIBLE_DYNAMIC_OPS=true
export TF_XLA_DECLUSTER_POSSIBLE_DYNAMIC_OPS=true
export TF_CPP_MIN_LOG_LEVEL=3

export DATA_DIR=/data/tensorflow/training/datasets/Criteo/cpu_preprocess_dlrm/binary_dataset/
