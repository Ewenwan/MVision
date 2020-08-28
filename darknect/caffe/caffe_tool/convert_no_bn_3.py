#coding: utf-8
#!/usr/bin/env python
'''
# usage:
# convbn2conv.py 
    --src_proto src_debn_net.prototxt \
    --src_model src_bn.caffemodel \
    --dst_proto dest_debn_net.prototxt \
    --dst_model dest_debn.caffemodel \
    --caffe_path /data/caffe/python
'''
'''
吸收模型文件caffemodel中的BN层参数

'''
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import sys
import argparse
import os.path as osp

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert conv+bn+scale to conv')
    # 原去除bn层的prototxt
    parser.add_argument('--src_proto', dest='src_prototxt',
                        help='prototxt file defining the source network',
                        default=None, type=str)
    # 原caffemodel
    parser.add_argument('--src_model', dest='src_caffemodel',
                        help='model to convert',
                        default=None, type=str)
    parser.add_argument('--dst_proto', dest='dst_prototxt',
                        help='prototxt file defining the destination network',
                        default=None, type=str)
    parser.add_argument('--dst_model', dest='dst_caffemodel',
                        help='dest caffemodel',
                        default='result.caffemodel', type=str)
    # caffe主目录路径
    parser.add_argument('--caffe_path',dest='caffe_path',
                        help='absolute path of caffe',
                        default='None',type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.src_prototxt is None or args.src_caffemodel is None or args.dst_prototxt is None:
        parser.print_help()
        sys.exit(1)

    return args

def add_path(path):
    """
    purpose: add path in sys path
    args:
        path: path to be added
    """
    if path not in sys.path:
        sys.path.insert(0,path)


args = parse_args()
#caffe_path = osp.join(args.caffe_path,'python')
add_path(args.caffe_path)

import caffe
import pprint
import time, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pdb


### 按名字查找某一层
def find_layer(name, net):
    for idx, n in enumerate(net._layer_names):
        if n == name:
            return net.layers[idx]
    return None

### 合并bn层 bn_layer 到前一层的 卷积层 conv_layer 中
def merge_bn(bn_layer, conv_layer):
    scale_factor = bn_layer.blobs[2].data[0]
    # 尺度参数
    if scale_factor != 0:
        scale_factor = 1.0 / scale_factor
    # 均值参数
    mean = bn_layer.blobs[0].data
    # 方差参数
    var = bn_layer.blobs[1].data

    mean *= scale_factor
    #### TODO
    eps = 1e-5  # how to get the parameter 'eps' of BatchNorm lyaer
    var = np.sqrt(var * scale_factor + eps)
    # 卷积层参数维度  和 BN层参数维度一致 
    assert conv_layer.blobs[0].data.shape[0] == mean.shape[0]
    # 卷积 weight
    blob0 = conv_layer.blobs[0].data
    # 卷积偏置 bias
    blob1 = conv_layer.blobs[1].data
    for fi in range(mean.shape[0]):
        blob0[fi] /= var[fi]
        blob1[fi] = (blob1[fi] - mean[fi]) / var[fi]

### 合并lbn层(bn+scale)  到前一层的 卷及参数中
def merge_lbn(lbn_layer, conv_layer):
    scale = lbn_layer.blobs[0].data
    shift = lbn_layer.blobs[1].data
    mean = lbn_layer.blobs[2].data
    var = lbn_layer.blobs[3].data
    # pdb.set_trace()
    scale = scale.reshape(scale.size)
    shift = shift.reshape(shift.size)
    mean = mean.reshape(mean.size)
    var = var.reshape(var.size)

    eps = 1e-5  # how to get the parameter 'eps' of BatchNorm lyaer

    W = conv_layer.blobs[0].data
    bias = conv_layer.blobs[1].data

    assert W.shape[0] == mean.size
    assert bias.size == mean.size

    alpha = scale / np.sqrt(var + eps);
    # 变化 bias
    conv_layer.blobs[1].data[...] = alpha * (bias - mean) + shift;
    # 变换 weight
    for fi in range(mean.shape[0]):
        W[fi] *= alpha[fi]

### 合并 缩放层 到前一层的 卷及参数中
def merge_scale(scale_layer, conv_layer):
    scale = scale_layer.blobs[0].data
    shift = None
    if len(scale_layer.blobs) == 2:
        shift = scale_layer.blobs[1].data

    assert conv_layer.blobs[0].data.shape[0] == scale.shape[0]

    blob0 = conv_layer.blobs[0].data
    #if shift is not None:
    # 有bug
    # 卷积层 的 bias_term为true  这里 blob1 肯定会存在
    blob1 = conv_layer.blobs[1].data
    for fi in range(scale.shape[0]):
        # weight 执行 缩放系数
        blob0[fi] *= scale[fi]
        # bias 执行 缩放+平移
        if shift is not None:
            blob1[fi] = blob1[fi] * scale[fi] + shift[fi]
        else:
            blob1[fi] = blob1[fi] * scale[fi]


if __name__ == '__main__':
    
    print('Called with args:')
    print(args)
    # cpu模式
    caffe.set_mode_cpu()
    # 原网络
    src_net = caffe.Net(args.src_prototxt, args.src_caffemodel, caffe.TEST)
    # 目标网络 无权重参数
    dst_net = caffe.Net(args.dst_prototxt, caffe.TEST)
    # 目标网络数据清零
    for layer in dst_net.layers:
        for b in layer.blobs:
            b.data[...] = 0

    prev_conv_layer = None
    for name, layer in zip(src_net._layer_names, src_net.layers):
        print(name)
        
        # 在目标网络中找到 name命名的层
        dst_layer = find_layer(name, dst_net)
        if dst_layer is not None:
            # 该层blob的数量
            blob_n = min(len(layer.blobs), len(dst_layer.blobs))
            # 设置 目标网络该层的数据
            for i in range(blob_n):
                dst_layer.blobs[i].data[...] = layer.blobs[i].data[...]
            
            #记录原网络前一层卷积层参数
            if dst_layer.type == "Convolution":
                prev_conv_layer = dst_layer
        
        # 目标网络中这一层已经删除，需要将其参数合并到该层的前继卷积层
        if dst_layer is None:
            if layer.type == "BatchNorm":
                if len(prev_conv_layer.blobs) != 2:
                    print("layer %s must have bias term" % name)
                    sys.exit(0)
                # 吸收BatchNorm层 layer 到前一层 卷积层 prev_conv_layer 中
                merge_bn(layer, prev_conv_layer)
                
            elif layer.type == 'Scale':
                if len(layer.blobs) == 2 and len(prev_conv_layer.blobs) != 2:
                    print("layer %s must have bias term" % name)
                    sys.exit(0)
                # 吸收 Scale 层 到前一层 卷积层 prev_conv_layer 中
                merge_scale(layer, prev_conv_layer)
            
            elif layer.type == "LBN":
                if len(prev_conv_layer.blobs) != 2:
                    print("layer %s must have bias term" % name)
                    sys.exit(0)
                # 吸收LBN(BatchNorm+Scale)层 到前一层 卷积层 prev_conv_layer 中
                merge_lbn(layer, prev_conv_layer)
         
            elif layer.type == "BN":
                if len(prev_conv_layer.blobs) != 2:
                    print("layer %s must have bias term" % name)
                    sys.exit(0)
                # 吸收LBN(BatchNorm+Scale)层 到前一层 卷积层 prev_conv_layer 中
                merge_lbn(layer, prev_conv_layer)

    dst_net.save(args.dst_caffemodel)
