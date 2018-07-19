#-*- coding:utf-8 -*-
#!/usr/bin/env python
import numpy as np
import cv2
import os, sys
# export PYTHONPATH=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov1_oldVer/python
sys.path.insert(0, '../../python/')
import caffe
GPU_ID = 1 # 0,1,2,3,4,5,6,7...
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

# 非极大值抑制输出
def nms(dets, thresh):
  # -------------------------
  # Pure Python NMS baseline.
  # Written by Ross Girshick
  # -------------------------
  # 坐标
  x1 = dets[:, 0] - dets[:, 2] / 2.#左上点坐标
  y1 = dets[:, 1] - dets[:, 3] / 2.
  x2 = dets[:, 0] + dets[:, 2] / 2.#右下点坐标
  y2 = dets[:, 1] + dets[:, 3] / 2.
  # 得分
  scores = dets[:, 4]
  # 区域面积
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  # 预测得分最大的框编号
  # 剩余集合 的编号
  order = scores.argsort()[::-1]
  keep = []#保留下来的边框
  while order.size > 0:
    i = order[0]# 得分最高的
    keep.append(i)
    # 计算最大的边框与剩余的边框的交并比
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h#交集
    # IOU = 交集/并集
    ovr = inter / (areas[i] + areas[order[1:]] - inter)
    # 剔除 IOU过高的框，既剔除重合度较高的边框
    inds = np.where(ovr <= thresh)[0]#保留下来的框id
    order = order[inds + 1]#保留下来的边框id
  return dets[np.require(keep), :]

#  对模型输出结果进行解码 
def parse_result(reg_out):
  num_classes = 20# 类别数量
  num_objects = 2 # 每个格子预测的边框数量
  side = 7        # 格子 
  locations = side ** 2# 格子总数
  # 每个格子对应6个参数 4个坐标参数 + 1个类别参数 + 1个准确度得分参数
  boxes = np.zeros((num_objects * locations, 6), dtype=np.float32)
  
  # 遍历每一个格子
  for i in range(locations):
    # 类别得分
    tmp_scores = reg_out[i:num_classes*locations:locations]
    # 最大类别概率 id
    max_class_ind = np.argsort(tmp_scores)[-1]
    # 最大预测概率
    max_prob = np.max(tmp_scores)
    obj_index = num_classes * locations + i
    # 最大类别概率得分
    obj_scores = max_prob * reg_out[obj_index:(obj_index+num_objects*locations):locations]
    # 坐标
    coor_index = (num_classes + num_objects) * locations + i
    # 每一个预测框
    for j in range(num_objects):
      boxes[i*num_objects+j][5] = max_class_ind# 类id
      boxes[i*num_objects+j][4] = obj_scores[j]#预测得分
      box_index = coor_index + j * 4 * locations# 坐标id
      boxes[i*num_objects+j][0] = (i % side + reg_out[box_index + 0 * locations]) / float(side)
      boxes[i*num_objects+j][1] = (i / side + reg_out[box_index + 1 * locations]) / float(side)
      boxes[i*num_objects+j][2] = reg_out[box_index + 2 * locations] ** 2
      boxes[i*num_objects+j][3] = reg_out[box_index + 3 * locations] ** 2
  # 非极大值抑制输出
  return nms(boxes, 0.7)

def show_boxes(im_path, boxes, thresh=0.5, show=0):
  import random
  r = lambda: random.randint(0,255)
  print boxes.shape
  # voc 数据集 20类
  classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
  im = cv2.imread(im_path)
  # 原图像尺寸
  ori_w = im.shape[1]
  ori_h = im.shape[0]
  # 对于每一个预测边框
  for box in boxes:
    if box[4] < thresh:# 0,1,2,3为坐标 4为得分 得分小于阈值的跳过
      continue
    print box
    color = (r(),r(),r())
    #box = box[:4]# 0,1,2,3为坐标  截取坐标
    # 左上点坐标
    xmin = max(0, int((box[0] - box[2] / 2.) * ori_w))
    ymin = max(0, int((box[1] - box[3] / 2.) * ori_h))
    # 右下点坐标
    xmax = min(ori_w - 1, int((box[0] + box[2] / 2.) * ori_w))
    ymax = min(ori_h - 1, int((box[1] + box[3] / 2.) * ori_h))
    # 这里颜色  需要变换 随机数
    # 画矩形      图像   左上点 右下点    颜色      粗细
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 3)
    # 显示标签  粗细值为负 表示填充矩形
    cv2.rectangle(im,(xmin,ymin-20),(xmax,ymin), color,-1)
    # cv2.putText()
    # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
    #cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)
    cv2.putText(im, classes[int(box[5])] + " : %.2f" % box[4],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
  # 显示图像
  #if show:
  cv2.imshow("out", im)
  # 保存图像 
  #else:
  cv2.imwrite("out.jpg", im)

# 检测主调函数
def det(model, im_path, show=0):
  '''forward processing'''
  # 图像预处理
  im = cv2.imread(im_path)
  # 变形
  im = cv2.resize(im, (448, 448))#宽，高，通道
  # 通道 宽高
  im = np.require(im.transpose((2, 0, 1)), dtype=np.float32)
  # 去均值
  im -= mean
  # caffe blobs数据
  model.blobs['data'].data[...] = im
  # 模型前向传播
  out_blobs = model.forward()
  # 获取模型结果
  reg_out = out_blobs["regression"]
  # 对结果解码，获取边框参数
  boxes = parse_result(reg_out[0])
  # 显示边框和类别
  show_boxes(im_path, boxes, 0.2)

if __name__=="__main__":
  net_proto = "./gnet_deploy.prototxt"
  #model_path = "./yolov1_models/yolov1_models_iter_26000.caffemodel"
  model_path = "./yolov1_models/yolov1_models_iter_32000.caffemodel"
  model = caffe.Net(net_proto, model_path, caffe.TEST)
  #if sys.argc > 1:
  im_path = sys.argv[1]
  #else:
  #im_path = './dog.jpg'
  
  det(model, im_path)#显示图像

