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

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

PARSER = argparse.ArgumentParser(description="UNet-3D")

# Estimator flags
PARSER.add_argument('--model_dir', required=True, type=str)
PARSER.add_argument('--exec_mode', choices=['train', 'evaluate', 'train_and_evaluate',
                                            'predict', 'debug_train', 'debug_predict'], type=str)

# Training flags
PARSER.add_argument('--benchmark', default=False, type=str2bool)
PARSER.add_argument('--max_steps', default=16000, type=int)
PARSER.add_argument('--learning_rate', default=0.0002, type=float)
PARSER.add_argument('--log_every', default=100, type=int)
PARSER.add_argument('--save_checkpoint_steps', default=1000, type=int)
PARSER.add_argument('--log_dir', type=str)
PARSER.add_argument('--loss', choices=['dice', 'ce', 'dice+ce'], default='dice+ce', type=str)
PARSER.add_argument('--warmup_steps', default=40, type=int)
PARSER.add_argument('--normalization', choices=['instancenorm', 'batchnorm', 'groupnorm'],
                    default='instancenorm', type=str)
PARSER.add_argument('--include_background', dest='include_background', action='store_true', default=False)
PARSER.add_argument('--resume_training', default=False, type=str2bool)
PARSER.add_argument('--use_horovod', default=False, type=str2bool)
PARSER.add_argument('--use_gpu', default=False, type=str2bool)
PARSER.add_argument('--use_amp', default=False, type=str2bool)
PARSER.add_argument('--use_performance', default=False, type=str2bool)
PARSER.add_argument('--use_profiler', default=False, type=str2bool)


# Augmentations
PARSER.add_argument('--augment', default=False, type=str2bool)

# Dataset flags
PARSER.add_argument('--data_dir', required=True, type=str)
PARSER.add_argument('--batch_size', default=1, type=int)
PARSER.add_argument('--fold', default=0, type=int)
PARSER.add_argument('--num_folds', default=5, type=int)

# Tensorflow configuration flags
PARSER.add_argument('--use_xla', '--xla', dest='use_xla', action='store_true', default=False)

