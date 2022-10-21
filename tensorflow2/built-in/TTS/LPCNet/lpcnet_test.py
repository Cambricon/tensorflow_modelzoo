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


from imp import load_module
import sys
import os
import importlib

cur_path = os.getcwd()
sys.path.append(cur_path + "/models/training_tf2")
lpcnet = importlib.import_module("mlu_lpcnet")

import numpy as np
from ulaw import ulaw2lin, lin2ulaw
import h5py

import argparse
from dataloader import LPCNetLoader
import lpcnet


def get_args():
    parser = argparse.ArgumentParser(description="Test an LPCNet model")

    parser.add_argument(
        "--checkpoint", metavar="<model>", help="trained model file (.h5)"
    )
    parser.add_argument(
        "--features", metavar="<features file>", help="binary features file (float32)"
    )
    parser.add_argument(
        "--output",
        metavar="<audio data file>",
        help="audio data file (16-bit raw 16 kHz PCM format)",
    )
    parser.add_argument(
        "--output_dir", metavar="<output dir>", help="directory for saving output"
    )

    args = parser.parse_args()
    return args


def create_output_dir(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    return


def construct_model(args):
    filename = args.checkpoint
    with h5py.File(filename, "r") as f:
        if "gru_cell" in f["model_weights"]["gru_a"]["gru_a"].keys():
            units = min(
                f["model_weights"]["gru_a"]["gru_a"]["gru_cell"][
                    "recurrent_kernel:0"
                ].shape
            )
            units2 = min(
                f["model_weights"]["gru_b"]["gru_b"]["gru_cell_1"][
                    "recurrent_kernel:0"
                ].shape
            )
        else:
            units = min(
                f["model_weights"]["gru_a"]["gru_a"]["recurrent_kernel:0"].shape
            )
            units2 = min(
                f["model_weights"]["gru_b"]["gru_b"]["recurrent_kernel:0"].shape
            )
        e2e = "rc2lpc" in f["model_weights"]

    model, enc, dec = lpcnet.new_lpcnet_model(
        training=False, rnn_units1=units, rnn_units2=units2, flag_e2e=e2e, batch_size=1
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return (model, enc, dec, e2e)


def load_weights(args, model):
    filename = args.checkpoint
    model.load_weights(filename)
    return model


def process_features(args, model):
    nb_frames = 1
    nb_features = 36
    feature_file = args.features
    frame_size = model.frame_size

    features = np.fromfile(feature_file, dtype="float32")
    features = np.resize(features, (-1, nb_features))
    feature_chunk_size = features.shape[0]
    features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))

    periods = (0.1 + 50 * features[:, :, 18:19] + 100).astype("int16")
    pcm_chunk_size = frame_size * feature_chunk_size
    pcm = np.zeros((nb_frames * pcm_chunk_size,))
    return (
        nb_frames,
        frame_size,
        nb_features,
        feature_chunk_size,
        features,
        pcm,
        periods,
    )


def model_infer(args, model, enc, dec, e2e):
    (
        nb_frames,
        frame_size,
        nb_features,
        feature_chunk_size,
        features,
        pcm,
        periods,
    ) = process_features(args, model)
    nb_used_features = model.nb_used_features
    order = 16
    skip = order + 1
    mem = 0
    coef = 0.85

    fexc = np.zeros((1, 1, 3), dtype="int16") + 128
    state1 = np.zeros((1, model.rnn_units1), dtype="float32")
    state2 = np.zeros((1, model.rnn_units2), dtype="float32")
    out_file = "{}/{}".format(args.output_dir, args.output)
    fout = open(out_file, "wb")

    for c in range(0, nb_frames):
        if not e2e:
            cfeat = enc.predict(
                [features[c : c + 1, :, :nb_used_features], periods[c : c + 1, :, :]]
            )
        else:
            cfeat, lpcs = enc.predict(
                [features[c : c + 1, :, :nb_used_features], periods[c : c + 1, :, :]]
            )
        for fr in range(0, feature_chunk_size):
            f = c * feature_chunk_size + fr
            if not e2e:
                a = features[c, fr, nb_features - order :]
            else:
                a = lpcs[c, fr]
            for i in range(skip, frame_size):
                pred = -sum(
                    a
                    * pcm[f * frame_size + i - 1 : f * frame_size + i - order - 1 : -1]
                )
                fexc[0, 0, 1] = lin2ulaw(pred)

                p, state1, state2 = dec.predict(
                    [fexc, cfeat[:, fr : fr + 1, :], state1, state2]
                )
                # Lower the temperature for voiced frames to reduce noisiness
                p *= np.power(p, np.maximum(0, 1.5 * features[c, fr, 19] - 0.5))
                p = p / (1e-18 + np.sum(p))
                # Cut off the tail of the remaining distribution
                p = np.maximum(p - 0.002, 0).astype("float64")
                p = p / (1e-8 + np.sum(p))

                fexc[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0, 0, :], 1))
                pcm[f * frame_size + i] = pred + ulaw2lin(fexc[0, 0, 2])
                fexc[0, 0, 0] = lin2ulaw(pcm[f * frame_size + i])
                mem = coef * mem + pcm[f * frame_size + i]
                np.array([np.round(mem)], dtype="int16").tofile(fout)
            skip = 0


def main(args):
    create_output_dir(args)
    model, enc, dec, e2e = construct_model(args)
    model = load_weights(args, model)
    model_infer(args, model, enc, dec, e2e)


if __name__ == "__main__":
    args = get_args()
    main(args)
