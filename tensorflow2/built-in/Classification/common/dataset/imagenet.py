from .base import DatasetBase, RawData
import tensorflow as tf


class ImageNetDataset(DatasetBase):
    def __init__(self, data_path=None, label_offset=None):
        super(ImageNetDataset, self).__init__(data_path)
        self.feature_map = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
            'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
        }
        self.label_offset = label_offset

    def decode(self, record):
        obj = tf.io.parse_single_example(record, self.feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        if self.label_offset is not None:
            label -= self.label_offset
        data = RawData()
        data.imageData = imgdata
        data.image_label = label
        return data


DATASET_DICT = dict()
DATASET_DICT["imagenet"] = ImageNetDataset(label_offset=1)

# Supporting 0 offset for classifier models such as mobilenet etc.
DATASET_DICT["imagenet_offset0"] = ImageNetDataset(label_offset=0)
