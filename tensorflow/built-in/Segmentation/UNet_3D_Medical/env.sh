export HOROVOD_STALL_SHUTDOWN_TIME_SECONDS=7200
export HOROVOD_STALL_CHECK_TIME_SECONDS=3600
export DATA_DIR=/data/tensorflow/training/datasets/MICCAI_BraTS_2019_Data_Training_Preprocess
UNET3D_HOME=${PWD}/UNet_3D_Medical/models
export PYTHONPATH=$(pwd):${UNET3D_HOME}:${PYTHONPATH}
