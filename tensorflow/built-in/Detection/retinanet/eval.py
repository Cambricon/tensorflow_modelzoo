# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main function to train various object detection models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import functools
import os
import pprint
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.optimizers import Adam, SGD
from models.modeling.hyperparams import params_dict
from models.utils import hyperparams_flags
from models.configs import factory as config_factory
from models.dataloader import input_reader
from models.dataloader import mode_keys as ModeKeys
from models.modeling import factory as model_factory

from models.cocotools import coco_eval

hyperparams_flags.initialize_common_flags()

flags.DEFINE_bool(
    'enable_xla',
    default=False,
    help='Enable XLA for GPU')

flags.DEFINE_string(
    'model', default='retinanet',
    help='Model to run: `retinanet` or `shapemask`.')

flags.DEFINE_string('training_file_pattern', None,
                    'Location of the train data.')

flags.DEFINE_string('eval_file_pattern', None, 'Location of ther eval data')

flags.DEFINE_string('val_json_file', None, 'Instances json file of the eval data')

flags.DEFINE_string(
    'checkpoint_path', None,
    'The checkpoint path to eval. Only used in eval_once mode.')

FLAGS = flags.FLAGS


def run(callbacks=None):
  params = config_factory.config_generator(FLAGS.model)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override(
      {
          'eval' : {
            'eval_file_pattern': FLAGS.eval_file_pattern,
            'val_json_file': FLAGS.val_json_file,
          }
      },
      is_strict=False)
  print("eval_file_pattern: ",params.eval.eval_file_pattern)
  print("val_file_pattern: ",params.eval.val_json_file)
  print("checkpoint_path: ",FLAGS.checkpoint_path)

  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info('Model Parameters: {}'.format(params_str))

  eval_input_fn = None
  eval_file_pattern = FLAGS.eval_file_pattern or params.eval.eval_file_pattern

  eval_input_fn = input_reader.InputFn(
    file_pattern=eval_file_pattern,
    params=params,
    mode=input_reader.ModeKeys.EVAL,
    batch_size=params.eval.batch_size,
    num_examples=params.eval.eval_samples)

  dataset = eval_input_fn()
  iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
  image, labels = iterator.get_next()

  model_builder = model_factory.model_generator(params)

  pred = model_builder.build_outputs(image, mode="eval")
  labels, outputs = model_builder.post_processing(labels, pred)

  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path) if tf.gfile.IsDirectory(FLAGS.checkpoint_path) else FLAGS.checkpoint_path
  print(latest_checkpoint)


  init_op = iterator.make_initializer(dataset)

  saver = tf.train.Saver()

  config = tf.ConfigProto()

  sess = tf.Session(config=config)

  saver.restore(sess, latest_checkpoint)
  sess.run(init_op)

  outputs_list = []
  for i in tqdm(range(int(5000 / params.eval.batch_size))):
      pred_out = sess.run(outputs)
      outputs_list.append([
          pred_out["source_id"],
          pred_out["detection_boxes"],
          pred_out["detection_scores"],
          pred_out["detection_classes"],
          pred_out["num_detections"],
          pred_out["image_info"],
      ])

  import pickle as pkl
  with open('eval.pkl', "wb") as f:
      pkl.dump(outputs_list, f)
  coco_eval(outputs_list, anno_file=FLAGS.val_json_file)


def main(argv):
  del argv  # Unused.

  run()


if __name__ == '__main__':
  app.run(main)
