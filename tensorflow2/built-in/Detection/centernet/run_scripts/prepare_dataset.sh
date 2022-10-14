cur_path=$(pwd)
work_dir="${cur_path}/.."
pushd "${work_dir}"

if [ ! -d "./data" ]; then

    if [ -z ${DATASET} ]; then
        export DATASET="/data/tensorflow/training/datasets/COCO17/tfrecords/"
    fi

    if [ -z ${DATASET_LABEL} ]; then
        export DATASET_LABEL="/data/tensorflow/training/datasets/tfrecord_coco/mscoco_label_map.pbtxt"
    fi
    COCO_DIR="./data/coco2017_tfrecords"
    if [ ! -d $COCO_DIR ]; then
        mkdir -p $COCO_DIR
        chmod 777 $COCO_DIR
        ln -sf $DATASET $COCO_DIR
        ln -sf $DATASET_LABEL $COCO_DIR
    fi
fi
popd
