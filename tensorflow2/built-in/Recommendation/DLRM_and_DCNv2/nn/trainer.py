# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import tensorflow as tf
import horovod.tensorflow as hvd
import embedding_table_plugin as etp

from absl import flags
FLAGS = flags.FLAGS
if FLAGS.mlu:
    from distributed_embeddings import dist_model_parallel as dmp
else:
    from distributed_embeddings.python.layers import dist_model_parallel as dmp

from .nn_utils import create_inputs_dict


class Trainer:
    def __init__(self, model, embedding_optimizer, mlp_optimizer, amp, lr_scheduler, tf_dataset_op, cpu):
        self.model = model
        self.embedding_optimizer = embedding_optimizer
        self.mlp_optimizer = mlp_optimizer
        self.amp = amp
        self.lr_scheduler = lr_scheduler
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        self.cpu = cpu
        self.tf_dataset_op = tf_dataset_op
        self.dataset_iter = iter(self.tf_dataset_op())

    def _weight_update(self, gradients):
        if self.amp:
            gradients = self.mlp_optimizer.get_unscaled_gradients(gradients)

        dense_gradients, dense_variables = [], []
        embedding_gradients, embedding_variables = [], []
        embedding_refs = set(v.ref() for v in self.model.sparse_model.trainable_variables)

        for var, grad in zip(self.model.trainable_variables, gradients):
            if var.ref() in embedding_refs:
                embedding_variables.append(var)
                embedding_gradients.append(grad)
            else:
                dense_variables.append(var)
                dense_gradients.append(grad)

        self.mlp_optimizer.apply_gradients(zip(dense_gradients, dense_variables))
        self.embedding_optimizer.apply_gradients(zip(embedding_gradients, embedding_variables))

    @tf.function
    def train_step(self):
        device = '/CPU:0' if self.cpu else '/MLU:0' if FLAGS.mlu else '/GPU:0'
        with tf.device(device):
            self.lr_scheduler()
            with tf.name_scope("dataloading"):
                (numerical_features, categorical_features), labels = self.dataset_iter.get_next()

            inputs = create_inputs_dict(numerical_features, categorical_features)
            with tf.GradientTape() as tape:
                predictions = self.model(inputs=inputs, training=True)
                unscaled_loss = self.bce(labels, predictions)
                # tf keras doesn't reduce the loss when using a Custom Training Loop
                unscaled_loss = tf.math.reduce_mean(unscaled_loss)
                scaled_loss = self.mlp_optimizer.get_scaled_loss(unscaled_loss) if self.amp else unscaled_loss

            # local embeddings should not communicate using AllReduce
            # TODO(xupengfei): remove the following code when cnhorovod supports register_local_source
            import horovod
            if horovod.__version__ < '0.27.0':
              all_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
              gradients = []
              for var, grad in zip(self.model.trainable_variables, all_gradients):
                if hasattr(var, 'de_local') or hvd.size() <= 1:
                  gradients.append(grad)
                else:
                  gradients.append(hvd.allreduce(grad, op=hvd.Average,
                                   compression=hvd.compression.NoneCompressor))
            else:
              if hvd.size() > 1:
                  tape = dmp.DistributedGradientTape(tape)
              gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

            self._weight_update(gradients)

            if hvd.size() > 1:
                # compute mean loss for all workers for reporting
                mean_loss = hvd.allreduce(unscaled_loss, name="mean_loss", op=hvd.Average)
            else:
                mean_loss = unscaled_loss

            return mean_loss
