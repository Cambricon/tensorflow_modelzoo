import tensorflow as tf
a = tf.ones(shape=[4, 224, 224, 3])
s = a.get_shape()
d = len(s)
if d == 2:
    reduce_dims = [0]
elif d == 4:
    reduce_dims = [0, 1, 2]
if d >= 2:
    min_a = tf.math.reduce_min(a, axis=reduce_dims)
else:
    min_a = a
print(min_a.get_shape())