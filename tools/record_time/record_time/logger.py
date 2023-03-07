import tensorflow as tf
import json
import numpy as np
import time

class TimeHistoryRecord(tf.keras.callbacks.Callback):
    """Use callback to record the time taken for each iteration."""
    def on_train_begin(self, logs={}):
        self.times = []

    def on_batch_begin(self, batch, logs=None):
        self.time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        self.times.append(time.time() - self.time_start)

class TimeHook(tf.estimator.SessionRunHook):
    """Use hooks to record the time taken for each iteration."""
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self._start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        self.times.append(current_time - self._start_time)

def write_json(summary_dir, global_batch_size, time_record):
    """Writes a summary text file to record fps."""
    if not tf.io.gfile.exists(summary_dir):
        tf.io.gfile.mkdir(summary_dir)

    if len(time_record) == 300:
        mean_time = np.mean(time_record[10:300])
    else:
        record_steps = len(time_record)/2 if len(time_record) <= 40 else len(time_record) - 40
        mean_time = np.mean(time_record[int(record_steps):])
    summary_dict = {"overall_stats":{"batch_size": global_batch_size, "throughput_mean": global_batch_size / mean_time, "e2e": mean_time}}
    summary_json = json.dumps(summary_dict)
    with open('{}/summary.json'.format(summary_dir), 'w') as json_file:
        json_file.write(summary_json)
        json_file.write('\n')
