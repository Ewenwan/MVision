# -*- coding: utf-8 -*-
# 常用函数等核心函数 预处理图像  后端处理图像  
# 在图片上显示检测结果  按照预测边框的得分进行排序
# 计算两个边框之间的交并比  非极大值抑制排除重复的边框
"""
Help functions for YOLOv2
"""
import random
import colorsys

import cv2
import numpy as np


############## 预处理图像        ##################
############## preprocess image ##################
def preprocess_image(image, image_size=(416, 416)):
    """Preprocess a image to inference"""
    image_cp = np.copy(image).astype(np.float32)# Float32格式
    # 图像变形固定尺寸 resize the image
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)# BGR 转成 RGB
    image_resized = cv2.resize(image_rgb, image_size)# 变形到固定尺寸
    # 归一化到 0~1 normalize
    image_normalized = image_resized.astype(np.float32) / 255.0# 除以最大值 255 -> 0~1
    # 扩展一个维度来存储 批次大小 expand the batch_size dim 416*416 -> 1*416*416
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded
    
############## 后端处理图像 #######################
def postprocess(bboxes, obj_probs, class_probs, image_shape=(416, 416),
                threshold=0.5):
    """post process the detection results 处理检测结果 """
    bboxes = np.reshape(bboxes, [-1, 4])# 边框值 0~1之间
    bboxes[:, 0::2] *= float(image_shape[1])# 乘上 图像大小 变成 实际大小 0 2 列 正比高 h
    bboxes[:, 1::2] *= float(image_shape[0])# 1 3列 正比 宽 W
    bboxes = bboxes.astype(np.int32)# 截取整数

    # 边框要在 图像大小之内 clip the bboxs
    bbox_ref = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    bboxes = bboxes_clip(bbox_ref, bboxes)
    # 预测的 目标/非目标 概率
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_inds = np.argmax(class_probs, axis=1)#类别
    class_probs = class_probs[np.arange(len(obj_probs)), class_inds]#类别概率
    scores = obj_probs * class_probs# 分数 = 目标/非目标 概率 * 类别概率

    # filter bboxes with scores > threshold
    keep_inds = scores > threshold# 得分大于阈值 的索引
    bboxes = bboxes[keep_inds]# 对应的得分较好的边框
    scores = scores[keep_inds]# 对应的得分
    class_inds = class_inds[keep_inds]# 对应的预测类型

    # 按照得分 排序sort top K
    class_inds, scores, bboxes = bboxes_sort(class_inds, scores, bboxes)
    # 非极大值抑制 排除 重叠度较大的预测框nms
    class_inds, scores, bboxes = bboxes_nms(class_inds, scores, bboxes)

    return bboxes, scores, class_inds# 最终的 边框 得分 类别索引
    
# 在图片上显示 检测结果
def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # for display
    ############################
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.)
                  for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)# 线条粗细
        # 画长方形 
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(imgcv, mess, text_loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv


############## process bboxes 边框要在 图像大小之内  ##################
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)#实际边框 中心点 宽 高
    bboxes = np.transpose(bboxes)#转置
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])#
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes
############## 按照预测边框的得分 进行排序 前 400个#################
def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]#类别索引排序
    scores = scores[idxes][:top_k]#得分排序
    bboxes = bboxes[idxes][:top_k]#最优预测边框排序
    return classes, scores, bboxes#类别 得分 边框

################ 计算 两个边框之间 的交并比 ########################
################ 交叠部分的面积， 并集   IOU = 交集/并集
def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w# 交叠部分的面积
    # 并集的面积 = S1+S2-交集.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    iou = int_vol / (vol1 + vol2 - int_vol)
    return iou
    
################# 非极大值抑制 排除重复的边框##########################
########## 输入为已排序的
########## 1. 选取得分较高的边框 在剩下的边框中 排除掉 与该边框 IOU较大的边框
########## 2. 再次选取得分较高的边框 按1的方法处理    直到边框被排除到没有
########## 3. 每次选择出来的边框 即为最优的边框
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)# 边框是否保留的标志
    for i in range(scores.size-1):
        if keep_bboxes[i]:#剩余的边框
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])#计算交并比
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]
