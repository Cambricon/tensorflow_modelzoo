# Copyright 2020 Huy Le Nguyen (@usimarit)
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

import os
import sys
from tqdm import tqdm
from absl import flags

cur_path = os.getcwd()
models_path = cur_path + "/models/"
sys.path.append(models_path)

from tensorflow_asr.utils import env_util
from tensorflow_asr.utils import file_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(models_path, "config.yml")
tf.keras.backend.clear_session()


def get_flags():
    flags.DEFINE_bool(
        "use_horovod", default=False, help="Whether to use horovod to train network.Although this is infer mode, we still needs this parameter in that some called scripts needs it."
    )
    flags.DEFINE_string(
        "config", default=DEFAULT_YAML, help="The file path of model configuration file"
    )
    flags.DEFINE_string(
        "saved",
        default=None,
        help="The file path of checkpoint file generated from train process",
    )
    flags.DEFINE_string("data_dir", default=None, help="Path to test data")
    flags.DEFINE_integer("batch_size", default=1, help="Test batch size.")
    flags.DEFINE_bool(
        "sentence_piece", default=False, help="Whether to use `SentencePiece` model"
    )
    flags.DEFINE_bool("subwords", default=True, help="Use subwords")
    flags.DEFINE_integer("device", default=0, help="Device's id to run test on.")
    flags.DEFINE_bool(
        "use_gpu", default=False, help="Whether to run training on GPU devices."
    )
    flags.DEFINE_string("output", default="test.tsv", help="Result filepath")
    flags.DEFINE_bool("get_rtf", default=False, help="Generate infer performance(RTF) data.")
    flags.DEFINE_bool("mxp", default=False, help="Enable mixed precision")
    flags.DEFINE_integer(
        "private_threadpool_size",
        default=4,
        help="If set, the dataset will use a private threadpool of the given size. "
        "The value 0 can be used to indicate that the threadpool size should be "
        "determined at runtime based on the number of available CPU cores.",
    )
    FLAGS = flags.FLAGS
    return FLAGS


def main(argv):
    FLAGS = get_flags()
    FLAGS(sys.argv)
    assert FLAGS.saved
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": FLAGS.mxp})

    env_util.setup_devices([FLAGS.device], gpu=FLAGS.use_gpu)

    from tensorflow_asr.configs.config import Config
    from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
    from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
    from tensorflow_asr.featurizers.text_featurizers import (
        SubwordFeaturizer,
        SentencePieceFeaturizer,
        CharFeaturizer,
    )
    from tensorflow_asr.models.transducer.conformer import Conformer
    from tensorflow_asr.utils import app_util

    config = Config(FLAGS.config)
    speech_featurizer = TFSpeechFeaturizer(config.speech_config)

    if FLAGS.sentence_piece:
        logger.info("Use SentencePiece ...")
        text_featurizer = SentencePieceFeaturizer(config.decoder_config)
    elif FLAGS.subwords:
        logger.info("Use subwords ...")
        text_featurizer = SubwordFeaturizer(config.decoder_config)
    else:
        logger.info("Use characters ...")
        text_featurizer = CharFeaturizer(config.decoder_config)

    tf.random.set_seed(0)

    if FLAGS.data_dir:
        config.learning_config.test_dataset_config.data_paths = [
            os.path.join(FLAGS.data_dir, i)
            for i in config.learning_config.test_dataset_config.data_paths
        ]
    test_dataset = ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.test_dataset_config),
    )

    # build model
    conformer = Conformer(
        **config.model_config, vocabulary_size=text_featurizer.num_classes
    )
    conformer.make(speech_featurizer.shape)
    conformer.load_weights(FLAGS.saved, by_name=True)
    conformer.summary(line_length=100)
    conformer.add_featurizers(speech_featurizer, text_featurizer)

    batch_size = FLAGS.batch_size or config.learning_config.running_config.batch_size
    test_data_loader = test_dataset.create(batch_size)
    if FLAGS.get_rtf:
        infer_duration = 0.0
        predict_duration = 0.0
        decode_duration = 0.0
        import time

    wav_duration = 0.0

    with file_util.save_file(file_util.preprocess_paths(FLAGS.output)) as filepath:
        infer_begin = time.time()
        results = conformer.predict(test_data_loader, verbose=1)
        infer_end = time.time()
        predict_duration += infer_end - infer_begin
        logger.info(f"Saving result to {FLAGS.output} ...")
        with open(filepath, "w") as openfile:
            openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
            progbar = tqdm(total=test_dataset.total_steps, unit="batch")
            for i, pred in enumerate(results):
                decode_begin = time.time()
                groundtruth, greedy, beamsearch = [x.decode("utf-8") for x in pred]
                decode_end = time.time()
                decode_duration += decode_end - decode_begin
                path, duration, _ = test_dataset.entries[i]
                wav_duration += float(duration)
                openfile.write(
                    f"{path}\t{duration}\t{groundtruth}\t{greedy}\t{beamsearch}\n"
                )
                progbar.update(1)
            progbar.close()
        infer_duration = predict_duration + decode_duration
        rtf = infer_duration*1.0/wav_duration
        logger.info(f"predict duration is:{predict_duration} ")
        logger.info(f"decode duration is:{decode_duration} ")
        logger.info(f"Infer duration is:{infer_duration} ")
        logger.info(f"Infer RTF is:{rtf} ")
        app_util.evaluate_results(filepath)


if __name__ == "__main__":
    tf.compat.v1.app.run()
