#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.02.04'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, callbacks
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import tf_lenet


# class MyModule(tf.Module):
#     def __init__(self, model):
#         self.model = model
#
#     @tf.function(input_signature=[])
#     def ckpt_write(self):
#         ckpt = tf.train.Checkpoint(self.model)
#         ckpt.write("./saved_ckpt")
#         return {"res": True}


x_train, y_train, x_test, y_test = tf_lenet.loadData()
model, opt = tf_lenet.createLeNet5Model(x_train)
model.compile(
   optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'],
)
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=0)
model.save("/tmp/tf_models/saved_model")
loaded = tf.saved_model.load('/tmp/tf_models/saved_model')
infer = loaded.signatures['serving_default']

f = tf.function(infer).get_concrete_function(conv2d_input=tf.TensorSpec(shape=[None, 32, 32, 1], dtype=tf.float32))
f2 = convert_variables_to_constants_v2(f)
graph_def = f2.graph.as_graph_def()

# write frozen graph (single file) to disk
with tf.io.gfile.GFile('../../rs/lenet_tf_rs/lenet5_frozen_graph.pb', 'wb') as f:
   f.write(graph_def.SerializeToString())

# my_module = MyModule(model)
# tf.saved_model.save(my_module, "./saved", signatures={"ckpt_write": my_module.ckpt_write})
# # my_module.ckpt_write()
#
# m = tf.keras.models.load_model("./saved")
# print(m)
# print(dir(m))
# m.ckpt_write()

# ckpt = tf.train.Checkpoint()
# read_m = ckpt.read("./saved_ckpt")
# print(read_m)
