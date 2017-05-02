#!/usr/bin/env python

import tensorflow as tf
import numpy

# preparing data
x = numpy.array([1.1, 2.1, 4.3, -1.2, -2.4, -3.5])
y = numpy.array([0.7, 1.0, 3.2, -1.1, -2.1, -3.4])

o = numpy.ones(x.shape)
w = numpy.ones([1, 2])

X  = numpy.array([x, o])
Y  = numpy.array([y])

x_data = tf.placeholder(tf.float32, shape=X.shape)
y_data = tf.placeholder(tf.float32, shape=Y.shape)

weight = tf.Variable(w, dtype=tf.float32)

# y = x*w
y_comp = tf.matmul(weight, x_data)

# build loss function
loss = tf.reduce_mean(tf.square(y_data - y_comp))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# initialize tensorflow session
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# training
for step in range(100):
     sess.run(train, feed_dict={ x_data: X, y_data: Y })
     print(step, sess.run(weight))

