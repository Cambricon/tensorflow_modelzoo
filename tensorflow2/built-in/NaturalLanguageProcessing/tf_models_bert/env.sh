export PRETRAINED_MODEL_PATH=/data/tensorflow/training/datasets/TensorFlow_Models_Bert/uncased_L-12_H-768_A-12
export SQUAD_DATA_PATH=/data/tensorflow/training/datasets/Bert/SQuAD
cur_path=$(pwd)
models_path="${cur_path}"/models/models
export PYTHONPATH=$PYTHONPATH:${models_path}
