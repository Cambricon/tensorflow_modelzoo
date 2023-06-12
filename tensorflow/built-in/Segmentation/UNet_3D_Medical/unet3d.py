# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import logging
import sys

sys.path.append("./models")
sys.path.append("./models/runtime")
sys.path.append("./models/model")
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from dataset.data_loader import Dataset, CLASSES
from runtime.hooks import get_hooks, ProfilingHook, TrainingHook
from runtime.arguments import PARSER
from runtime.setup import prepare_model_dir, build_estimator, set_flags, get_logger


def parse_evaluation_results(result):
    data = {CLASSES[i]: result[CLASSES[i]] for i in range(len(CLASSES))}
    data['MeanDice'] = sum([result[CLASSES[i]] for i in range(len(CLASSES))]) / len(CLASSES)
    data['WholeTumor'] = result['WholeTumor']
    return data


def main():
    tf.get_logger().setLevel(logging.ERROR)
    params = PARSER.parse_args()
    set_flags(params)
    hvd.init()
    model_dir = prepare_model_dir(params)
    logger = get_logger(params)

    if params.use_performance and params.use_profiler:
        raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

    if params.use_performance:
        from record_time import TimeHook, write_json
        global TimeHook
        global write_json

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold_idx=params.fold,
                      n_folds=params.num_folds,
                      params=params)

    estimator = build_estimator(params=params, model_dir=model_dir)

    if params.use_performance or params.use_profiler:
        max_steps = params.max_steps
    else:
        max_steps = params.max_steps // (1 if params.benchmark else hvd.size())

    if 'train' in params.exec_mode:
        training_hooks = get_hooks(params, logger)
        if params.use_performance and hvd.rank() == 0:
            time_hooks = TimeHook()
            training_hooks.append(time_hooks)
        if params.use_profiler and hvd.rank() == 0:
            timeline_hook = tf.estimator.ProfilerHook(save_steps=5, output_dir='./profiler')
            training_hooks.append(timeline_hook)
        estimator.train(
            input_fn=dataset.train_fn,
            steps=max_steps,
            hooks=training_hooks)
        if params.use_performance and hvd.rank() == 0:
            write_json("summary", params.batch_size * hvd.size(), time_hooks.times)

    if 'evaluate' in params.exec_mode:
        need_record_flag = False
        real_batch_size = 0
        if params.use_performance:
            if params.use_horovod:
                if hvd.rank() == 0:
                    need_record_flag = True
                    real_batch_size = params.batch_size * hvd.size()
            else:
                need_record_flag = True
                real_batch_size = params.batch_size
        if need_record_flag != False:
            eval_hooks = get_hooks(params, logger)
            time_hooks = TimeHook()
            eval_hooks.append(time_hooks)
            result = estimator.evaluate(
                input_fn=dataset.eval_fn, steps=dataset.eval_size, hooks=eval_hooks
            )
        else:
            result = estimator.evaluate(
                input_fn=dataset.eval_fn, steps=dataset.eval_size
            )

        data = parse_evaluation_results(result)
        if hvd.rank() == 0:
            logger.log(step=(), data=data)
        if need_record_flag == True:
            write_json("summary", real_batch_size, time_hooks.times)

        #result = estimator.evaluate(input_fn=dataset.eval_fn, steps=dataset.eval_size)
        #data = parse_evaluation_results(result)
        #if hvd.rank() == 0:
        #    logger.log(step=(), data=data)

    if 'predict' == params.exec_mode:
        inference_hooks = get_hooks(params, logger)
        if hvd.rank() == 0:
            count = 1 if not params.benchmark else 2 * params.warmup_steps * params.batch_size // dataset.test_size
            predictions = estimator.predict(
                input_fn=lambda: dataset.test_fn(count=count,
                                                 drop_remainder=params.benchmark), hooks=inference_hooks)

            for idx, p in enumerate(predictions):
                volume = p['predictions']
                if not params.benchmark:
                    np.save(os.path.join(params.model_dir, "vol_{}.npy".format(idx)), volume)

    if 'debug_train' == params.exec_mode:
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        if hvd.rank() == 0:
            hooks += [TrainingHook(log_every=params.log_every,
                                   logger=logger,
                                   tensor_names=['total_loss_ref:0']),
                      ProfilingHook(warmup_steps=params.warmup_steps,
                                    global_batch_size=hvd.size() * params.batch_size,
                                    logger=logger,
                                    mode='train')]

        estimator.train(
            input_fn=dataset.synth_train_fn,
            steps=max_steps,
            hooks=hooks)

    if 'debug_predict' == params.exec_mode:
        if hvd.rank() == 0:
            hooks = [ProfilingHook(warmup_steps=params.warmup_steps,
                                   global_batch_size=params.batch_size,
                                   logger=logger,
                                   mode='inference')]
            count = 2 * params.warmup_steps
            predictions = estimator.predict(input_fn=lambda: dataset.synth_predict_fn(count=count),
                                            hooks=hooks)
            for p in predictions:
                _ = p['predictions']


if __name__ == '__main__':
    main()

