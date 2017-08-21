# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
numerictensor= tf.constant(1.0)

graph_location = "/home/fabricio/_tasks/201704_grupo_pesquisa_ilb/tutorial_tensorflow/tmp"#tempfile.mkdtemp()

sess = tf.Session()

summary = tf.summary.scalar('test', numerictensor)
#tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
train_writer = tf.summary.FileWriter(graph_location, graph=sess.graph)
#train_writer.add_summary(summary, global_step=None)
train_writer.flush()

print('Saving graph to: %s' % graph_location)
#train_writer = tf.summary.FileWriter(graph_location)
#train_writer.add_graph(tf.get_default_graph())

print(sess.run(hello))