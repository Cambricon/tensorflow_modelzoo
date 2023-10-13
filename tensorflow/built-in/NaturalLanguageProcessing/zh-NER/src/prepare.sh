data_dir="data_path"
if [ ! -d "$data_dir" ]; then
  cp -rf ${DATASET_DIR}/$data_dir .
fi
model_dir="data_path_save"
if [ ! -d "$model_dir" ]; then
  cp -rf ${DATASET_DIR}/$model_dir .
fi
