#-*- coding: utf-8 -*-
# 解码网络输出 到实际的表示方式
"""
Detection ops for Yolov2
"""

import tensorflow as tf
import numpy as np

# 默认80个类别
def decode(detection_feat, feat_sizes=(13, 13), num_classes=80,
           anchors=None):
    """decode from the detection feature"""
    H, W = feat_sizes# 最后 特征图的 尺寸 13*13格子数量
    num_anchors = len(anchors)# 每个格子预测的 边框数量 
    detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors,
                                        num_classes + 5])

    bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, 0:2])# 边框中心点 相对于所在格子 左上点 的偏移的比例
    bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])# 
    obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])# 物体 
    class_probs = tf.nn.softmax(detetion_results[:, :, :, 5:])

    anchors = tf.constant(anchors, dtype=tf.float32)

    height_ind = tf.range(H, dtype=tf.float32)
    width_ind = tf.range(W, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    x_offset = tf.reshape(x_offset, [1, -1, 1])
    y_offset = tf.reshape(y_offset, [1, -1, 1])

    # decode
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5

    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h,
                       bbox_x + bbox_w, bbox_y + bbox_h], axis=3)

    return bboxes, obj_probs, class_probs
