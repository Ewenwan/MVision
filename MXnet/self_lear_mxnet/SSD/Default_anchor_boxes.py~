#!/usr/bin/env python
#-*- coding:utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior##MultiBoxPrior产生预设框

n = 40
# 输入形状: batch × channel × height × weight
x = nd.random_uniform(shape=(1, 3, n, n))  
##               图像    n 个预设尺寸      m 个预设的长宽比    输出为 n+m-1 个方框
y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])

## 取位于 (20,20) 像素点的第一个预设框
# box的格式为 (x_min, y_min, x_max, y_max) 且为比例
boxes = y.reshape((n, n, -1, 4))
print('The first anchor box at row 21, column 21:', boxes[20, 20, 0, :])

import matplotlib.pyplot as plt
#"""convert an anchor box to a matplotlib rectangle"""
def box_to_rect(box, color, linewidth=3):
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]),
        fill=False, edgecolor=color, linewidth=linewidth)
colors = ['blue', 'green', 'red', 'black', 'magenta']# 3+3-1=5个
plt.imshow(nd.ones((n, n, 3)).asnumpy())
anchors = boxes[20, 20, :, :]
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
plt.show()
