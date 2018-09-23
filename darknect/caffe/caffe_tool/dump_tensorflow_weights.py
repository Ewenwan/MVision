# -*- coding:utf-8 -*-
# 需要安装 tensorflow
# pip3 install tensorflow==1.4.0
# pip3 install tensorflow-gpu==1.4.0
import tensorflow as tf
#import cv2
import numpy as np
import os

# temsflow中的 网络图模型
def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'rb') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='',return_elements=[])

graph_create("ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb")
save_dir = ('output')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with tf.Session() as sess:
    tensors = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
    for t in tensors:
      if ('weights' in t.name \
             or 'bias' in t.name \
             or 'moving_variance' in t.name \
             or 'moving_mean' in t.name \
             or 'beta' in t.name \
             or 'gamma' in t.name \
             or 'BatchNorm/batchnorm/sub' in t.name \
             or 'BatchNorm/batchnorm/mul' in t.name) \
             and ('sub' in t.name or 'mul/' in t.name or 'read' in t.name): 
         ts = tf.get_default_graph().get_tensor_by_name(t.name + ":0")
         data = ts.eval()
         #print(t.name)
         names = t.name.split('/')
         layer_name = None
         paratype = None
         if 'BatchNorm/batchnorm/sub' in t.name or 'biases' in t.name:
              paratype = 'biases'
         elif 'BatchNorm/batchnorm/mul' in t.name:
              paratype = 'weights_scale'
         elif 'weights' in t.name:
              paratype = 'weights'
         elif 'moving_variance' in t.name:
              paratype = 'bn_moving_variance'
         elif 'moving_mean' in t.name:
              paratype = 'bn_moving_mean'
         elif 'beta' in t.name:
              paratype = 'beta'
         elif 'gamma' in t.name:
              paratype = 'gamma'

         if names[2] == 'Conv' or names[2] == 'Conv_1':
              layer_name = names[2]
         elif 'expanded_conv' in names[2]:
              layer_name = names[2].replace('expanded_', '') + '_' + names[3]
         elif  'layer_19' in names[2]:
              substr = names[2].split('_')
              layer_name = 'layer_19_' + substr[2] + '_' + substr[4]
              if 'depthwise' in names[2]:
                  layer_name += "_depthwise"
         elif  'BoxPredictor' in names[0]:
              layer_name = names[0] + '_' + names[1]
         output_name = layer_name + "_"  + paratype
         # print output_name
         print(output_name)
         #print ts.get_shape()
         
         if len(data.shape) == 4:
             caffe_weights = data.transpose(3, 2, 0, 1)
             origin_shape = caffe_weights.shape
             boxes = 0
             if 'depthwise' not in output_name:
                 if output_name.find('BoxEncodingPredictor') != -1:
                     boxes = caffe_weights.shape[0] // 4 # 整数除法
                 elif output_name.find('ClassPredictor') != -1:
                     boxes = caffe_weights.shape[0] // 91

                 if output_name.find('BoxEncodingPredictor') != -1:
                     tmp = caffe_weights.reshape(boxes, 4, -1).copy()
                     new_weights = np.zeros(tmp.shape, dtype=np.float32)
                     #tf order:    [y, x, h, w]
                     #caffe order: [x, y, w, h]
                     if 'BoxPredictor_0/BoxEncodingPredictor/weights' in t.name:
                         #caffe first box layer [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)]
                         #tf first box layer    [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)]
                         #adjust the box by weights and bias change
                         new_weights[:, 0] = tmp[:, 1] * 0.5
                         new_weights[:, 1] = tmp[:, 0] * 0.5
                     else:
                         new_weights[:, 0] = tmp[:, 1]
                         new_weights[:, 1] = tmp[:, 0]
                     new_weights[:, 2] = tmp[:, 3]
                     new_weights[:, 3] = tmp[:, 2]
                     caffe_weights = new_weights.reshape(origin_shape).copy()
                 if output_name.find('BoxEncodingPredictor') != -1 or \
                     output_name.find('ClassPredictor') != -1:
                     tmp = caffe_weights.reshape(boxes, -1).copy()
                     new_weights = np.zeros(tmp.shape, dtype=np.float32)
                     #tf aspect ratio:   [1, 2, 3, 0.5, 0.333333333, 1]
                     #caffe aspect ratio:[1, 1, 2, 3, 0.5, 0.333333333]
                     if boxes == 6:
                         new_weights[0] = tmp[0]
                         new_weights[1] = tmp[5]
                         new_weights[2] = tmp[1]
                         new_weights[3] = tmp[2]
                         new_weights[4] = tmp[3]
                         new_weights[5] = tmp[4]
                         caffe_weights = new_weights.reshape(origin_shape).copy()
             caffe_weights.tofile(os.path.join(save_dir, output_name + '.dat'))
             # print caffe_weights.shape
             print(caffe_weights.shape)
         else:
             caffe_bias = data
             boxes = 0
             if 'depthwise' not in output_name:
                 if output_name.find('BoxEncodingPredictor') != -1:
                     boxes = caffe_bias.shape[0] // 4 # 整数除法
                 elif output_name.find('ClassPredictor') != -1:
                     boxes = caffe_bias.shape[0] // 91
                 if output_name.find('BoxEncodingPredictor') != -1:
                     tmp = caffe_bias.reshape(boxes, 4).copy()
                     new_bias = np.zeros(tmp.shape, dtype=np.float32)
                     new_bias[:, 0] = tmp[:, 1]
                     new_bias[:, 1] = tmp[:, 0]
                     new_bias[:, 2] = tmp[:, 3]
                     new_bias[:, 3] = tmp[:, 2]
                     caffe_bias = new_bias.flatten().copy()

                 if output_name.find('BoxEncodingPredictor') != -1 or \
                     output_name.find('ClassPredictor') != -1:
                     tmp = caffe_bias.reshape(boxes, -1).copy()
                     new_bias = np.zeros(tmp.shape, dtype=np.float32)
                     if boxes == 6:
                         new_bias[0] = tmp[0]
                         new_bias[1] = tmp[5]
                         new_bias[2] = tmp[1]
                         new_bias[3] = tmp[2]
                         new_bias[4] = tmp[3]
                         new_bias[5] = tmp[4]
                         caffe_bias = new_bias.flatten()
                     elif 'BoxPredictor_0/BoxEncodingPredictor/biases' in t.name:
                         #caffe first box layer [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)]
                         #tf first box layer    [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)]
                         #adjust the box by weights and bias change
                         new_bias[0,:2] = tmp[0,:2] * 0.5
                         new_bias[0,2] = tmp[0,2] + (np.log(0.5) / 0.2)
                         new_bias[0,3] = tmp[0,3] + (np.log(0.5) / 0.2)
                         new_bias[1] = tmp[1]
                         new_bias[2] = tmp[2]
                         caffe_bias = new_bias.flatten()
                 # print caffe_bias.shape
                 print(caffe_bias.shape)
             caffe_bias.tofile(os.path.join(save_dir, output_name + '.dat'))
