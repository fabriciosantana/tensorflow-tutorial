#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:22:57 2017

@author: fabricio
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    print("teste")
    #mnist is a lightweight class which stores the training, validation, and testing sets as NumPy arrays
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print("teste2")
    #It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation
    x = tf.placeholder(tf.float32, [None, 784])
    
    #A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    #softmaxn regression
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.matmul(x, W) + b
                    
    #We try to minimize that error, and the smaller the error margin, the better our model is
    #cost or loss: cross-entropy -->  represents how far off our model is from our desired outcome
    #our predictions are for describing the truth
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()
    
    
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      #print(_)
      
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
