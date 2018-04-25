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
    H, W = feat_sizes
    C = num_classes
    B = len(anchors)
    anchors = tf.constant(anchors, dtype=tf.float32)
    anchors = tf.reshape(anchors, [1, 1, B, 2])

    sprob, sconf, snoob, scoor = scales  # the scales for different parts

    _coords = targets["coords"]  # ground truth [-1, H*W, B, 4]
    _probs = targets["probs"]    # class probability [-1, H*W, B, C] one hot
    _confs = targets["confs"]    # 1 for object, 0 for background, [-1, H*W, B]

    # decode the net output
    predictions = tf.reshape(predictions, [-1, H, W, B, (5 + C)])
    coords = predictions[:, :, :, :, 0:4]   # t_x, t_y, t_w, t_h
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    coords_xy = tf.nn.sigmoid(coords[:, :, :, 0:2])  # (0, 1) relative cell top left
    coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * anchors /
                        np.reshape([W, H], [1, 1, 1, 2])) # sqrt of w, h (0, 1)
    coords = tf.concat([coords_xy, coords_wh], axis=3)  # [batch_size, H*W, B, 4]

    confs = tf.nn.sigmoid(predictions[:, :, :, :, 4])  # object confidence
    confs = tf.reshape(confs, [-1, H*W, B, 1])

    probs = tf.nn.softmax(predictions[:, :, :, :, 5:])  # class probability
    probs = tf.reshape(probs, [-1, H*W, B, C])

    preds = tf.concat([coords, confs, probs], axis=3)  # [-1, H*W, B, (4+1+C)]

    # match ground truths with anchors (predictions in fact)
    # assign ground truths to the predictions with the best IOU (select 1 among 5 anchors)
    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    up_left, down_right = centers - (wh * 0.5), centers + (wh * 0.5)

    # the ground truth
    _wh = tf.pow(_coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
    _centers = _coords[:, :, :, 0:2]
    _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)

    # compute IOU
    inter_upleft = tf.maximum(up_left, _up_left)
    inter_downright = tf.minimum(down_right, _down_right)
    inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)
    intersects = inter_wh[:, :, :, 0] * inter_wh[:, :, :, 1]
    ious = tf.truediv(intersects, areas + _areas - intersects)

    best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=2, keep_dims=True))
    best_iou_mask = tf.cast(best_iou_mask, tf.float32)
    mask = best_iou_mask * _confs  # [-1, H*W, B]
    mask = tf.expand_dims(mask, -1)  # [-1, H*W, B, 1]

    # compute weight terms
    confs_w = snoob * (1 - mask) + sconf * mask
    coords_w = scoor * mask
    probs_w = sprob * mask
    weights = tf.concat([coords_w, confs_w, probs_w], axis=3)

    truths = tf.concat([_coords, tf.expand_dims(_confs, -1), _probs], 3)

    loss = tf.pow(preds - truths, 2) * weights
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    loss = 0.5 * tf.reduce_mean(loss)
    return loss
