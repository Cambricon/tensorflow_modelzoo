# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base import DatasetBase, RawData
import os
import sys
cur_path = os.getcwd()
sys.path.append(cur_path+ "/models/model")
print(sys.path)
import tokenization
sys.path.append(cur_path+ "/../transformer/models")
from official.nlp.data import squad_lib
import tensorflow.compat.v1 as tf

class SquadDataset(DatasetBase):
    def __init__(self):
        super(SquadDataset,self).__init__(None)
        self.max_seq_length=512
        self.max_query_length=64
        self.doc_stride = 128
        self.version_2_with_negative=False
        self.batchsize=None
        self.folder_path=None
        self.vocab_suffix="cased_L-12_H-768_A-12/vocab.txt"
        self.json_file_suffix="SQuAD/dev-v1.1.json"
        self.do_lower_case=False
        self.eval_examples=None
        self.eval_features=None
        self.tfrecord_gen_path="./tmp_record.tf_record"

        self.feature_map={
            'unique_ids': tf.io.FixedLenFeature([],dtype=tf.int64),
            'input_ids': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64),
            'input_mask': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64),
            'segment_ids': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64)
        }

    def fetch_validation(self):
        return  self.tfrecord_gen_path

    def valid_dataset_path(self):
        if os.path.isdir(self.folder_path):
            return os.path.isfile(os.path.join(self.folder_path,self.json_file_suffix)) and \
                    os.path.isfile(os.path.join(self.folder_path,self.vocab_suffix))
        else:
            return False

    def set_dataset_folder(self,folder_path):
        self.folder_path=folder_path
        flag=self.valid_dataset_path()
        if not flag:
            raise ValueError("dataset folder:{} is not valid!".format(folder_path))

    def get_json_path(self):
        return os.path.join(self.folder_path,self.json_file_suffix)

    def initialize(self):
        tokenizer = tokenization.FullTokenizer(
                                    vocab_file=os.path.join(self.folder_path,self.vocab_suffix),
                                    do_lower_case=self.do_lower_case)
        self.eval_examples = squad_lib.read_squad_examples(
                        input_file=os.path.join(self.folder_path,self.json_file_suffix),
                        is_training=False,
                        version_2_with_negative=self.version_2_with_negative)

        eval_writer = squad_lib.FeatureWriter(
            filename= self.tfrecord_gen_path,
            is_training=False)
        self.eval_features = []

        def _append_feature(feature, is_padding):
            if not is_padding:
                self.eval_features.append(feature)
            eval_writer.process_feature(feature)

        kwargs = dict(
            examples=self.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            output_fn=_append_feature,
            batch_size=self.batchsize)

        _ = squad_lib.convert_examples_to_features(**kwargs)
        eval_writer.close()

        logging.info('***** Running predictions *****')
        logging.info('  Num orig examples = %d', len(self.eval_examples))
        logging.info('  Num split examples = %d', len(self.eval_features))
        logging.info('  Batch size = %d', self.batchsize)

    def set_batchsize(self,batchsize):
        self.batchsize=batchsize

    def decode(self,record)-> RawData:
        obj = tf.io.parse_single_example(record, self.feature_map)
        res={
            "input_mask":tf.cast(obj["input_mask"],tf.int32),
            "input_ids":tf.cast(obj["input_ids"],tf.int32),
            "unique_ids": tf.cast(obj["unique_ids"],tf.int32),
            "segment_ids": tf.cast(obj["segment_ids"],tf.int32)
        }
        return res
