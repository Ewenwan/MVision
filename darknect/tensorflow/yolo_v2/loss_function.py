#-*- coding: utf-8 -*-
# yolo-v2 loss/cost/error function 损失/代价/误差函数
"""
Loss function for YOLOv2
"""

import numpy as np
import tensorflow as tf

def compute_loss(predictions, targets, anchors, scales, num_classes=20, feat_sizes=(13, 13)):
    """
    Compute the loss of Yolov2 for training
    """
    H, W = feat_sizes#最后 特征图的 尺寸 13*13格子数量
    C = num_classes# 总类别数量
    B = len(anchors)# 每个格子预测的 边框数量
    anchors = tf.constant(anchors, dtype=tf.float32)# 先验框
    anchors = tf.reshape(anchors, [1, 1, B, 2])#包含两个参数 框的 宽和高  对于图像大小的比例

    sprob, sconf, snoob, scoor = scales  # the scales for different parts

    _coords = targets["coords"]  # 坐标值 ground truth [-1, H*W, B, 4] 13*13个格子 每个格子预测5个边框 每个边框 4个参数
    _probs = targets["probs"]    # class probability [-1, H*W, B, C]  13*13个格子 每个格子预测20个类比的物体的概率
    _confs = targets["confs"]    # 1 for object, 0 for background, [-1, H*W, B] 边框的目标/背景 概率

    # 解码网络输出 到实际的表示方式 decode the net output
    predictions = tf.reshape(predictions, [-1, H, W, B, (5 + C)])# 变形到形象化尺寸
    coords = predictions[:, :, :, :, 0:4]   # t_x, t_y, t_w, t_h# 边框参数 坐标中心 宽  高
    coords = tf.reshape(coords, [-1, H*W, B, 4])# 变形 13*13个格子 5个边框  4个参数
    coords_xy = tf.nn.sigmoid(coords[:, :, :, 0:2])  # 中心点相对于格子   (0, 1) relative cell top left 左上角的偏移 比例
    coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * anchors /
                        np.reshape([W, H], [1, 1, 1, 2])) # sqrt of w, h (0, 1) 得到的为 宽高的sqrt 值
    coords = tf.concat([coords_xy, coords_wh], axis=3)  # [batch_size, H*W, B, 4]

    confs = tf.nn.sigmoid(predictions[:, :, :, :, 4])  # 物体置信度 object confidence 0/1
    confs = tf.reshape(confs, [-1, H*W, B, 1])

    probs = tf.nn.softmax(predictions[:, :, :, :, 5:])  # 类别概率 class probability改了
    probs = tf.reshape(probs, [-1, H*W, B, C])

    preds = tf.concat([coords, confs, probs], axis=3)  # [-1, H*W, B, (4+1+C)]

    # 预测输出值 match ground truths with anchors (predictions in fact)
    # assign ground truths to the predictions with the best IOU (select 1 among 5 anchors)
    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])# sqrt 后 平方得到原值 得到相对于特征图的 大小
    areas = wh[:, :, :, 0] * wh[:, :, :, 1]# 区域面积
    centers = coords[:, :, :, 0:2]#中心点
    up_left, down_right = centers - (wh * 0.5), centers + (wh * 0.5)# 得到预测边框 上左上点 和 右下点

    # 标记值 the ground truth
    _wh = tf.pow(_coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])# 得到相对于特征图的 大小 标记值
    _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]#区域面积
    _centers = _coords[:, :, :, 0:2]#中心点
    _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)# 得到 标记边框 上左上点 和 右下点

    # 计算 交并比 compute IOU
    inter_upleft = tf.maximum(up_left, _up_left)
    inter_downright = tf.minimum(down_right, _down_right)
    inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)#交叠区域
    intersects = inter_wh[:, :, :, 0] * inter_wh[:, :, :, 1]#交叠区域面积
    ious = tf.truediv(intersects, areas + _areas - intersects)#iou 交集/并集

    best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=2, keep_dims=True))
    best_iou_mask = tf.cast(best_iou_mask, tf.float32)#
    mask = best_iou_mask * _confs  # [-1, H*W, B] iou*物体置信度
    mask = tf.expand_dims(mask, -1)  # [-1, H*W, B, 1]#  

    # 计算各项 loss 的权重compute weight terms
    confs_w = snoob * (1 - mask) + sconf * mask#置信度误差 权重
    coords_w = scoor * mask# 坐标误差 权重
    probs_w = sprob * mask# 类别概率误差 权重
    weights = tf.concat([coords_w, confs_w, probs_w], axis=3)# 各项 loss 的权重

    truths = tf.concat([_coords, tf.expand_dims(_confs, -1), _probs], 3)

    loss = tf.pow(preds - truths, 2) * weights#计算各项 loss * loss权重
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])# 平方和 
    loss = 0.5 * tf.reduce_mean(loss)#均值
    return loss
