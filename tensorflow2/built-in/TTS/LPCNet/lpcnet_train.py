#!/usr/bin/python3
"""Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# Train an LPCNet model

import argparse
import sys
import os
import importlib
from tabnanny import verbose

cur_path = os.getcwd()
sys.path.append(cur_path + "/models/training_tf2")
lpcnet = importlib.import_module("mlu_lpcnet")
from dataloader import LPCNetLoader

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib

import tensorflow as tf
from tf_funcs import *
from lossfuncs import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Train an LPCNet model")

    parser.add_argument(
        "--features", metavar="<features file>", help="binary features file (float32)"
    )
    parser.add_argument(
        "--data", metavar="<audio data file>", help="binary audio data file (uint8)"
    )
    parser.add_argument("--output", metavar="<output>", help="trained model file (.h5)")
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--quantize", metavar="<input weights>", help="quantize model")
    group1.add_argument(
        "--retrain", metavar="<input weights>", help="continue training model"
    )
    group1.add_argument(
        "--finetune", metavar="<finetune model>", help="finetune training model file"
    )
    parser.add_argument(
        "--start_epoch",
        metavar="<epoch>",
        default=0,
        type=int,
        help="start epoch for finetune training model",
    )
    parser.add_argument(
        "--steps_per_epoch",
        metavar="<steps_per_epoc>",
        type=int,
        default=0,
        help="number of epochs to train for finetune",
    )
    parser.add_argument(
        "--density",
        metavar="<global density>",
        type=float,
        help="average density of the recurrent weights (default 0.1)",
    )
    parser.add_argument(
        "--density-split",
        nargs=3,
        metavar=("<update>", "<reset>", "<state>"),
        type=float,
        help="density of each recurrent gate (default 0.05, 0.05, 0.2)",
    )
    parser.add_argument(
        "--grub-density",
        metavar="<global GRU B density>",
        type=float,
        help="average density of the recurrent weights (default 1.0)",
    )
    parser.add_argument(
        "--grub-density-split",
        nargs=3,
        metavar=("<update>", "<reset>", "<state>"),
        type=float,
        help="density of each GRU B input gate (default 1.0, 1.0, 1.0)",
    )
    parser.add_argument(
        "--grua-size",
        metavar="<units>",
        default=384,
        type=int,
        help="number of units in GRU A (default 384)",
    )
    parser.add_argument(
        "--grub-size",
        metavar="<units>",
        default=16,
        type=int,
        help="number of units in GRU B (default 16)",
    )
    parser.add_argument(
        "--epochs",
        metavar="<epochs>",
        default=120,
        type=int,
        help="number of epochs to train for (default 120)",
    )
    parser.add_argument(
        "--batch_size",
        metavar="<batch size>",
        default=128,
        type=int,
        help="batch size to use (default 128)",
    )
    parser.add_argument(
        "--end2end",
        dest="flag_e2e",
        action="store_true",
        help="Enable end-to-end training (with differentiable LPC computation",
    )
    parser.add_argument(
        "--lr", metavar="<learning rate>", type=float, help="learning rate"
    )
    parser.add_argument(
        "--decay", metavar="<decay>", type=float, help="learning rate decay"
    )
    parser.add_argument(
        "--gamma",
        metavar="<gamma>",
        type=float,
        help="adjust u-law compensation (default 2.0, should not be less than 1.0)",
    )
    parser.add_argument(
        "--model_dir", metavar="<model dir>", help="directory for training model files"
    )
    parser.add_argument(
        "--num_gpus",
        metavar="<devices>",
        default=0,
        type=int,
        help="number of gpu devices",
    )
    parser.add_argument(
        "--num_mlus",
        metavar="<devices>",
        default=1,
        type=int,
        help="number of mlu devices",
    )
    parser.add_argument(
        "--use_amp",
        metavar="<mixed precision>",
        default=False,
        type=str2bool,
        help="whether to use mixed precision",
    )
    parser.add_argument(
        "--enable_tensorboard",
        metavar="<tensorboard>",
        type=str2bool,
        help="If use tensorboard, please set True.",
    )
    parser.add_argument(
        "--use_horovod",
        metavar="<use_horovod>",
        type=str2bool,
        help="If use horovod, please set True.",
    )
    parser.add_argument(
        "--use_performance",
        metavar="<use_performance>",
        type=str2bool,
        help="Use performance tools",
    )

    args = parser.parse_args()
    return args


def get_density(args):
    density = (0.05, 0.05, 0.2)
    if args.density_split is not None:
        density = args.density_split
    elif args.density is not None:
        density = [0.5 * args.density, 0.5 * args.density, 2.0 * args.density]

    return density


def get_grub_density(args):
    grub_density = (1.0, 1.0, 1.0)
    if args.grub_density_split is not None:
        grub_density = args.grub_density_split
    elif args.grub_density is not None:
        grub_density = [
            0.5 * args.grub_density,
            0.5 * args.grub_density,
            2.0 * args.grub_density,
        ]

    return grub_density


def get_gamma(args):
    gamma = 2.0 if args.gamma is None else args.gamma
    return gamma


def get_lr_and_decay(args):
    if args.quantize:
        lr = 0.00003
        decay = 0
    else:
        lr = 0.001
        decay = 2.5e-5
    if args.lr is not None:
        lr = args.lr
    if args.decay is not None:
        decay = args.decay
    return (lr, decay)


def get_opt(args):
    lr, decay = get_lr_and_decay(args)
    if args.use_horovod:
        opt = tf.keras.optimizers.legacy.Adam(lr, decay=decay, beta_2=0.99)
        opt = hvd.DistributedOptimizer(opt, sparse_as_dense=True)
    else:
        opt = tf.keras.optimizers.legacy.Adam(lr, decay=decay, beta_2=0.99)

    if args.use_amp:
        policy = tf.keras.mixed_precision.Policy(
            'mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    return opt


def get_rank_and_size(args):
    if args.use_horovod:
        rank = hvd.rank()
        size = hvd.size()
    else:
        rank = size = 1
    return (rank, size)


def init_horovod(args):
    if args.use_horovod:
        global hvd
        hvd.init()

        if (args.num_gpus > 0) and (args.num_mlus == 0):
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
        elif (args.num_gpus == 0) and (args.num_mlus > 0):
            mlus = tf.config.experimental.list_physical_devices("MLU")
            if mlus:
                tf.config.experimental.set_visible_devices(
                    mlus[hvd.local_rank()], "MLU"
                )
        else:
            raise ValueError(
                "Horovod can only be used when only one of gpu or mlu is greater than 0"
            )
    return


def create_model_dir(args):
    if (args.use_horovod and hvd.rank() == 0) or (not args.use_horovod):
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
    return


def get_strategy(args):
    local_devices = get_available_devs(args.num_gpus, args.num_mlus)
    strategy = tf.compat.v2.distribute.MirroredStrategy(local_devices)
    return strategy


def get_global_batch_size(args):
    strategy = get_strategy(args)
    batch_size = args.batch_size
    if args.use_horovod:
        global_batch_size = batch_size * hvd.size()
    else:
        global_batch_size = batch_size * strategy.num_replicas_in_sync

    return global_batch_size


def get_verbose(args):
    verbose = (
        1 if ((args.use_horovod and hvd.rank() == 0) or (not args.use_horovod)) else 0
    )
    return verbose


def construct_model(args):
    opt = get_opt(args)
    gamma = get_gamma(args)

    finetune = args.finetune is not None
    flag_e2e = args.flag_e2e
    batch_size = args.batch_size
    quantize = args.quantize

    if not args.use_horovod:
        strategy = get_strategy(args)

        with strategy.scope():
            if finetune:
                tf.compat.v1.logging.info(
                    "Load a fine-tuning model file from {}".format(args.finetune)
                )
                model = lpcnet.load_model(args.finetune)
            else:
                tf.compat.v1.logging.info("Create new lpcnet model.")
                model, _, _ = lpcnet.new_lpcnet_model(
                    rnn_units1=args.grua_size,
                    rnn_units2=args.grub_size,
                    batch_size=batch_size,
                    training=True,
                    quantize=quantize,
                    flag_e2e=flag_e2e,
                )
            if not flag_e2e:
                model.compile(
                    optimizer=opt,
                    loss="sparse_categorical_crossentropy",
                    metrics="sparse_categorical_crossentropy",
                )
            else:
                model.compile(
                    optimizer=opt,
                    loss=interp_mulaw(gamma=gamma),
                    metrics=[
                        metric_cel,
                        metric_icel,
                        metric_exc_sd,
                        metric_oginterploss,
                    ],
                )
            model.summary()
    else:
        if finetune:
            tf.compat.v1.logging.info(
                "Load a fine-tuning model file from {}".format(args.finetune)
            )
            model = lpcnet.load_model(args.finetune)
        else:
            tf.compat.v1.logging.info("Create new lpcnet model.")
            model, _, _ = lpcnet.new_lpcnet_model(
                rnn_units1=args.grua_size,
                rnn_units2=args.grub_size,
                batch_size=batch_size,
                training=True,
                quantize=quantize,
                flag_e2e=flag_e2e,
            )
        if not flag_e2e:
            model.compile(
                optimizer=opt,
                loss="sparse_categorical_crossentropy",
                metrics="sparse_categorical_crossentropy",
            )
        else:
            model.compile(
                optimizer=opt,
                loss=interp_mulaw(gamma=gamma),
                metrics=[metric_cel, metric_icel, metric_exc_sd, metric_oginterploss],
            )
        model.summary()

    return model


def load_weights(args, model):
    finetune = args.finetune is not None
    quantize = args.quantize is not None
    retrain = args.retrain is not None

    if args.quantize:
        input_model = args.quantize
    if retrain:
        input_model = args.retrain
    if quantize or retrain:
        model.load_weights(input_model)
    return model


def proc_input_data(args, model):
    feature_file = args.features
    pcm_file = args.data  # 16 bit unsigned short PCM samples
    frame_size = model.frame_size
    nb_features = 36
    nb_used_features = model.nb_used_features
    feature_chunk_size = 15
    pcm_chunk_size = frame_size * feature_chunk_size
    batch_size = args.batch_size
    # u for unquantised, load 16 bit PCM samples and convert to mu-law

    data = np.memmap(pcm_file, dtype="uint8", mode="r")
    nb_frames = (len(data) // (4 * pcm_chunk_size) - 1) // batch_size * batch_size

    features = np.memmap(feature_file, dtype="float32", mode="r")

    # limit to discrete number of frames
    data = data[4 * 2 * frame_size :]
    data = data[: nb_frames * 4 * pcm_chunk_size]
    data = np.reshape(data, (nb_frames, pcm_chunk_size, 4))

    sizeof = features.strides[-1]
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(nb_frames, feature_chunk_size + 4, nb_features),
        strides=(
            feature_chunk_size * nb_features * sizeof,
            nb_features * sizeof,
            sizeof,
        ),
    )
    features = features[:, :, :nb_used_features]

    periods = (0.1 + 50 * features[:, :, 18:19] + 100).astype("int16")

    return (data, features, periods)


def get_lpcloader(args, model):
    data, features, periods = proc_input_data(args, model)
    rank, size = get_rank_and_size(args)
    loader = LPCNetLoader(data, features, periods, args.batch_size, rank, size)
    return loader


def instantiated_sparsify(args):
    density = get_density(args)
    quantize = args.quantize is not None
    retrain = args.retrain is not None
    finetune = args.finetune is not None
    if quantize or retrain or finetune:
        if quantize:
            sparsify = lpcnet.Sparsify(10000, 30000, 100, density, quantize=True)
        else:
            sparsify = lpcnet.Sparsify(0, 0, 1, density)
    else:
        sparsify = lpcnet.Sparsify(2000, 40000, 400, density)
    return sparsify


def instantiated_grub_sparsify(args):
    grub_density = get_grub_density(args)
    quantize = args.quantize is not None
    retrain = args.retrain is not None
    finetune = args.finetune is not None
    if quantize or retrain or finetune:
        if quantize:
            grub_sparsify = lpcnet.SparsifyGRUB(
                10000, 30000, 100, args.grua_size, grub_density, quantize=True
            )
        else:
            grub_sparsify = lpcnet.SparsifyGRUB(0, 0, 1, args.grua_size, grub_density)
    else:
        grub_sparsify = lpcnet.SparsifyGRUB(
            2000, 40000, 400, args.grua_size, grub_density
        )
    return grub_sparsify


def dump_ckpt(args):
    checkpoint = ModelCheckpoint(
        "{}/{}_{}_{}.h5".format(
            args.model_dir, args.output, args.grua_size, "{epoch:02d}"
        )
    )
    return checkpoint


def generate_callbacks(args):
    sparsify = instantiated_sparsify(args)
    grub_sparsify = instantiated_grub_sparsify(args)
    callbacks = [sparsify, grub_sparsify]
    checkpoint = dump_ckpt(args)

    if args.use_horovod:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    if (args.use_horovod and hvd.rank() == 0) or (not args.use_horovod):
        callbacks.append(checkpoint)
        if args.use_performance:
            global time_callback
            callbacks.append(time_callback)

    if args.enable_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, profile_batch=2)
        callbacks.append(tensorboard_callback)

    return callbacks


def train_model(args, loader, model):
    verbose = get_verbose(args)
    callbacks = generate_callbacks(args)

    finetune = args.finetune is not None
    nb_epochs = args.epochs

    if finetune:
        model.fit(
            loader,
            initial_epoch=args.start_epoch,
            epochs=nb_epochs,
            validation_split=0.0,
            callbacks=callbacks,
            steps_per_epoch=args.steps_per_epoch,
            verbose=verbose,
        )
    else:
        model.fit(
            loader,
            epochs=nb_epochs,
            validation_split=0.0,
            callbacks=callbacks,
            steps_per_epoch=None if args.steps_per_epoch == 0 else args.steps_per_epoch,
            verbose=verbose,
        )
    return


def output_summary(args):
    if args.use_performance:
        if (args.use_horovod and hvd.rank() == 0) or (not args.use_horovod):
            global_batch_size = get_global_batch_size(args)
            global time_callback
            write_json("summary", global_batch_size, time_callback.times)
    return


def check_devs_num(device_type="MLU"):
    local_device_protos = device_lib.list_local_devices()
    devs_list = [x.name for x in local_device_protos if x.device_type == device_type]
    return len(devs_list)


def get_available_devs(num_gpus, num_mlus):
    assert num_gpus <= check_devs_num("GPU")
    assert num_mlus <= check_devs_num("MLU")

    if (num_gpus == 0) and (num_mlus == 0):
        devices = ["device:CPU:0"]
    elif (num_gpus > 0) and (num_mlus == 0):
        devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    elif (num_gpus == 0) and (num_mlus > 0):
        devices = ["device:MLU:%d" % i for i in range(num_mlus)]
    else:
        raise ValueError("Only one of mlu and gpu can be greater than 0")
    return devices


def main(args):
    init_horovod(args)
    create_model_dir(args)
    model = construct_model(args)
    loader = get_lpcloader(args, model)
    model = load_weights(args, model)
    train_model(args, loader, model)
    output_summary(args)


if __name__ == "__main__":
    args = get_args()
    if args.use_horovod:
        import horovod.tensorflow.keras as hvd
    if args.use_performance:
        from record_time import TimeHistoryRecord, write_json

        time_callback = TimeHistoryRecord()
    main(args)
