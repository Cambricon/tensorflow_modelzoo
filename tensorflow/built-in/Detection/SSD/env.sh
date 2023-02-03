export DATASET="/data/tensorflow/training/datasets/COCO17/tfrecords/"
export DATASET_LABEL="/data/tensorflow/training/datasets/tfrecord_coco/mscoco_label_map.pbtxt"
export COCO_DIR="./data/coco2017_tfrecords"
if [ ! -d $COCO_DIR ]; then
    mkdir -p $COCO_DIR
    chmod 777 $COCO_DIR
    ln -sf $DATASET $COCO_DIR
    ln -sf $DATASET_LABEL $COCO_DIR
fi

