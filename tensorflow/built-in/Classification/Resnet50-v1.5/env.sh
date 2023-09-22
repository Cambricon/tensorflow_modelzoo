CLASSIFICATION_HOME=$(pwd)/models/
export PYTHONPATH=${CLASSIFICATION_HOME}:$PYTHONPATH
export DATA_DIR=/data/tensorflow/training/datasets/ILSVRC2012/ilsvrc12_tfrecord
export DATA_IDX_DIR=/data/tensorflow/training/datasets/dali_index_dir