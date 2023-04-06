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

from decimal import HAVE_THREADS
import os
import sys
import math

cur_path = os.getcwd()
models_path = cur_path + "/models/"
sys.path.append(models_path)

from tensorflow_asr.utils import env_util
import tensorflow as tf
from keras.distribute import worker_training_state
from absl import flags
from tensorflow_asr.configs.config import Config


logger = env_util.setup_environment()
DEFAULT_YAML = os.path.join(models_path, "config.yml")
tf.keras.backend.clear_session()


class DummyContextManager(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def get_flags():
    flags.DEFINE_bool(
        "use_horovod", default=False, help="Whether to use horovod to train network"
    )

    flags.DEFINE_bool("use_profiler", default=False, help="Use profiler")

    flags.DEFINE_bool("use_performance", default=False, help="Use performance tools")

    flags.DEFINE_bool("use_gpu", False, "Whether to run training on GPU devices.")

    flags.DEFINE_string(
        "config", default=DEFAULT_YAML, help="The file path of model configuration file"
    )

    flags.DEFINE_bool("tfrecords", default=False, help="Whether to use tfrecords")

    flags.DEFINE_bool(
        "sentence_piece", default=False, help="Whether to use `SentencePiece` model"
    )

    flags.DEFINE_bool("subwords", default=True, help="Use subwords")

    flags.DEFINE_integer("batch_size", default=0, help="The number of batch_size.")

    flags.DEFINE_integer("steps", default=0, help="The number of steps.")

    flags.DEFINE_integer(
        "spx", default=1, help="Steps per execution for maximizing performance"
    )

    flags.DEFINE_string(
        "metadata", default=None, help="Path to file containing metadata"
    )

    flags.DEFINE_string("data_dir", default=None, help="Path to train/eval data")

    flags.DEFINE_bool("static_length", default=False, help="Use static lengths")

    flags.DEFINE_integer(
        "num_workers",
        default=1,
        help="When num_workers > 1, training uses "
        "MultiWorkerMirroredStrategy. When num_workers = 1 it uses "
        "MirroredStrategy.",
    )

    flags.DEFINE_bool("skip_eval", default=False, help="Whether to skip_eval per epoch")
    flags.DEFINE_bool("mxp", default=False, help="Enable mixed precision")

    flags.DEFINE_string("pretrained", default=None, help="Path to pretrained model")

    FLAGS = flags.FLAGS
    return FLAGS


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, backup_dir):
        self.backup_dir = backup_dir

    def on_train_begin(self, logs=None):
        self.model._training_state = worker_training_state.WorkerTrainingState(
            self.model, self.backup_dir
        )
        self._training_state = self.model._training_state
        self._training_state.restore()


def get_strategy(FLAGS):
    if FLAGS.use_horovod:
        strategy = None
    else:
        devices = [i for i in range(FLAGS.num_workers)]
        strategy = env_util.setup_strategy(devices, gpu=FLAGS.use_gpu)
    return strategy


def get_strategy_scope(FLAGS):
    if FLAGS.use_horovod:
        strategy_scope = DummyContextManager()
    else:
        strategy = get_strategy(FLAGS)
        strategy_scope = strategy.scope()
    return strategy_scope


def init_horovod(FLAGS):
    global hvd
    hvd.init()
    if FLAGS.use_gpu:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    else:
        mlus = tf.config.experimental.list_physical_devices("MLU")
        if mlus:
            tf.config.experimental.set_visible_devices(mlus[hvd.local_rank()], "MLU")

    return


def get_train_data_loader(
    FLAGS, bsz_tuple, speech_featurizer, text_featurizer, dataset_config
):
    train_dataset = get_train_dataset(
        FLAGS, speech_featurizer, text_featurizer, dataset_config
    )
    if FLAGS.use_horovod:
        # use batch_size
        train_data_loader = train_dataset.create(bsz_tuple[0])
    else:
        # use global_batch_size
        train_data_loader = train_dataset.create(bsz_tuple[1])

    train_dataset_total_steps = train_dataset.total_steps
    return (train_data_loader, train_dataset_total_steps)


def get_eval_data_loader(
    FLAGS, bsz_tuple, speech_featurizer, text_featurizer, dataset_config
):
    eval_dataset = get_eval_dataset(
        FLAGS, speech_featurizer, text_featurizer, dataset_config
    )
    eval_dataset_total_steps = 0
    global hvd
    if FLAGS.skip_eval or (FLAGS.use_horovod and hvd.rank() != 0):
        eval_data_loader = None
    else:
        if FLAGS.use_horovod:
            # use batch_size
            eval_data_loader = eval_dataset.create(bsz_tuple[0])
        else:
            # use global_batch_size
            eval_data_loader = eval_dataset.create(bsz_tuple[1])
    if eval_data_loader:
        eval_dataset_total_steps = eval_dataset.total_steps
    return (eval_data_loader, eval_dataset_total_steps)


def get_batch_size(FLAGS, config):
    batch_size = 0
    if FLAGS.batch_size > 0:
        batch_size = FLAGS.batch_size
    else:
        batch_size = config.learning_config.running_config.batch_size
    return batch_size


def get_global_batch_size(FLAGS, batch_size):
    if FLAGS.use_horovod:
        global hvd
        global_batch_size = batch_size * hvd.size()
    else:
        strategy = get_strategy(FLAGS)
        global_batch_size = batch_size * strategy.num_replicas_in_sync
    return global_batch_size


def get_dataset(FLAGS, speech_featurizer, text_featurizer, dataset_config):

    if FLAGS.tfrecords:
        cur_dataset = asr_dataset.ASRTFRecordDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(dataset_config),
            indefinite=True
        )
    else:
        cur_dataset = asr_dataset.ASRSliceDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(dataset_config),
            indefinite=True
        )
    return cur_dataset


def get_train_dataset(FLAGS, speech_featurizer, text_featurizer, dataset_config):
    train_dataset = get_dataset(
        FLAGS, speech_featurizer, text_featurizer, dataset_config
    )
    train_dataset.load_metadata(FLAGS.metadata)
    return train_dataset


def get_eval_dataset(FLAGS, speech_featurizer, text_featurizer, dataset_config):
    eval_dataset = get_dataset(
        FLAGS, speech_featurizer, text_featurizer, dataset_config
    )
    eval_dataset.load_metadata(FLAGS.metadata)
    return eval_dataset


def get_config(FLAGS):
    config = Config(FLAGS.config)
    if FLAGS.data_dir:
        config.decoder_config["corpus_files"] = [
            os.path.join(FLAGS.data_dir, i)
            for i in config.decoder_config["corpus_files"]
        ]
        config.learning_config.train_dataset_config.data_paths = [
            os.path.join(FLAGS.data_dir, i)
            for i in config.learning_config.train_dataset_config.data_paths
        ]
        config.learning_config.eval_dataset_config.data_paths = [
            os.path.join(FLAGS.data_dir, i)
            for i in config.learning_config.eval_dataset_config.data_paths
        ]
    return config


def get_text_featurizer(FLAGS, config):
    if FLAGS.sentence_piece:
        logger.info("Loading SentencePiece model ...")
        text_featurizer = text_featurizers.SentencePieceFeaturizer(
            config.decoder_config
        )
    elif FLAGS.subwords:
        logger.info("Loading subwords ...")
        text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
    else:
        logger.info("Use characters ...")
        text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)
    return text_featurizer


def get_speech_featurizer(FLAGS, config):
    speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)
    return speech_featurizer


def get_featurizer(FLAGS, config):
    speech_featurizer = get_speech_featurizer(FLAGS, config)
    text_featurizer = get_text_featurizer(FLAGS, config)
    return (speech_featurizer, text_featurizer)


def check_or_make_ckpt_dir(FLAGS, config):
    ckpt_head, ckpt_tail = os.path.split(
        config.learning_config.running_config.checkpoint["filepath"]
    )
    if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
        if not os.path.exists(ckpt_head):
            os.makedirs(ckpt_head)


def construct_model(FLAGS, config, speech_featurizer, text_featurizer, bsz_tuple, strategy_scope):
    batch_size = bsz_tuple[0]
    global_batch_size = bsz_tuple[1]
    if not FLAGS.static_length:
        speech_featurizer.reset_length()
        text_featurizer.reset_length()

    with strategy_scope:
        # build model
        conformer = Conformer(
            **config.model_config, vocabulary_size=text_featurizer.num_classes
        )
        conformer.make(
            speech_featurizer.shape,
            prediction_shape=text_featurizer.prepand_shape,
            batch_size=batch_size if FLAGS.use_horovod else global_batch_size,
        )
        if FLAGS.pretrained:
            if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
                conformer.load_weights(
                    FLAGS.pretrained, by_name=True, skip_mismatch=True
                )
        conformer.summary(line_length=100)
        optimizer = tf.keras.optimizers.Adam(
            TransformerSchedule(
                d_model=conformer.dmodel,
                warmup_steps=config.learning_config.optimizer_config.pop(
                    "warmup_steps", 10000
                ),
                max_lr=(0.00625 * global_batch_size / math.sqrt(conformer.dmodel)),
            ),
            **config.learning_config.optimizer_config
        )
        conformer.compile(
            optimizer=optimizer,
            experimental_steps_per_execution=FLAGS.spx,
            global_batch_size=global_batch_size,
            blank=text_featurizer.blank,
        )
        return conformer


def get_callbacks(FLAGS, config):
    global hvd
    if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                **config.learning_config.running_config.checkpoint
            ),
            tf.keras.callbacks.experimental.BackupAndRestore(
                config.learning_config.running_config.states_dir
            ),
        ]
        if FLAGS.use_profiler:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    **config.learning_config.running_config.tensorboard
                )
            )
        if FLAGS.use_performance:
            global time_callback
            callbacks.append(time_callback)
    else:
        callbacks = [CustomCallback(config.learning_config.running_config.states_dir)]
    if FLAGS.use_horovod:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    return callbacks


def train_model(FLAGS, config, strategy_scope):
    speech_featurizer, text_featurizer = get_featurizer(FLAGS, config)
    batch_size = get_batch_size(FLAGS, config)
    global_batch_size = get_global_batch_size(FLAGS, batch_size)
    bsz_tuple = (batch_size, global_batch_size)

    train_data_loader, train_dataset_total_steps = get_train_data_loader(
        FLAGS,
        bsz_tuple,
        speech_featurizer,
        text_featurizer,
        config.learning_config.train_dataset_config,
    )
    eval_data_loader, eval_dataset_total_steps = get_eval_data_loader(
        FLAGS,
        bsz_tuple,
        speech_featurizer,
        text_featurizer,
        config.learning_config.eval_dataset_config,
    )

    conformer = construct_model(
        FLAGS, config, speech_featurizer, text_featurizer, bsz_tuple, strategy_scope
    )

    # prepare for train
    callbacks = get_callbacks(FLAGS, config)
    epochs = 1 if FLAGS.steps > 0 else config.learning_config.running_config.num_epochs
    validation_steps = eval_dataset_total_steps if eval_data_loader else None
    steps_per_epoch = FLAGS.steps if FLAGS.steps > 0 else train_dataset_total_steps
    # start to train model
    conformer.fit(
        train_data_loader,
        epochs=epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    if FLAGS.use_performance:
        if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
            global time_callback
            global write_json
            write_json("summary", global_batch_size, time_callback.times)


def main(argv):
    FLAGS = get_flags()
    FLAGS(sys.argv)
    if FLAGS.use_performance:
        from record_time import TimeHistoryRecord, write_json

        global time_callback
        time_callback = TimeHistoryRecord()
        global write_json
    if FLAGS.use_horovod:
        import horovod.tensorflow.keras as hvd

        global hvd
        init_horovod(FLAGS)

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": FLAGS.mxp})
    config = get_config(FLAGS)
    strategy_scope = get_strategy_scope(FLAGS)

    from tensorflow_asr.datasets import asr_dataset
    from tensorflow_asr.featurizers import speech_featurizers, text_featurizers
    from tensorflow_asr.models.transducer.conformer import Conformer
    from tensorflow_asr.optimizers.schedules import TransformerSchedule

    global asr_dataset
    global speech_featurizers
    global text_featurizers
    global Conformer
    global TransformerSchedule
    check_or_make_ckpt_dir(FLAGS, config)
    train_model(FLAGS, config, strategy_scope)


if __name__ == "__main__":
    tf.compat.v1.app.run()
