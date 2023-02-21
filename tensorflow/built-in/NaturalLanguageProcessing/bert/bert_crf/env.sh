function check_python_version() {
  PYTHON_VERSION=`python --version 2>&1 | awk -F '[ .]' '{print $2"."$3}'`
  MINIMUM_REQUIRES=3.7
  if [[ $PYTHON_VERSION < $MINIMUM_REQUIRES ]]; then
    echo "[ERROR]: python version must be greater than $MINIMUM_REQUIRES,"
    echo "         but now python version is $PYTHON_VERSION"
    exit 1
  fi
}
check_python_version
export PYTHONPATH=$(pwd)/models/:${PYTHONPATH}
export DATA_DIR=/data/tensorflow/training/datasets/Bert_crf/
