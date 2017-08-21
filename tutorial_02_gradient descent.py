#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:22:13 2017

@author: fabricio
"""

import tensorflow as tf

sess = tf.Session()

#linear model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

#variable initialization
init = tf.global_variables_initializer()
sess.run(init)

#loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
