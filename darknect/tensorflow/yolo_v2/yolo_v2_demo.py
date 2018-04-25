#-*- coding: utf-8 -*-
# yolo-v2 调用示例
"""
Demo for yolov2 yolov2 示例
"""

import numpy as np # numpy
import tensorflow as tf # tensorflow
import cv2 # opencv
from PIL import Image# Python Imaging Library Python平台 上的图像处理标准库  pip install pillow

from model import darknet# 模型 model darknet
from decode_output import decode # 网输出结果 解码处理
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names# 聚类出来的  先验框 长宽 coco数据集 类别名称


input_size = (416, 416)# 检测图像大小 416*416 按 32步长增减
image_file = "./images/car.jpg"
image = cv2.imread(image_file)# 读取文件
image_shape = image.shape[:2]# 图像长宽
image_cp = preprocess_image(image, input_size)
"""
image = Image.open(image_file)
image_cp = image.resize(input_size, Image.BICUBIC)
image_cp = np.array(image_cp, dtype=np.float32)/255.0# 像素归一化 0~1
image_cp = np.expand_dims(image_cp, 0)
#print(image_cp)
"""


images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])# 输入图像
detection_feat = darknet(images)# 检测特征
feat_sizes = input_size[0] // 32, input_size[1] // 32# 经过32步长stride，最后特征图的尺寸为原图像尺寸/32
detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)# 解码输出检测结果
'''
保存和加载模型
Tensorflow 针对这一需求提供了Saver类
1. Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。
    Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值。
2. 只要提供一个计数器，当计数器触发时，Saver类可以自动的生成checkpoint文件。
    这让我们可以在训练过程中保存多个中间结果。
    例如，我们可以保存每一步训练的结果。
3. 为了避免填满整个磁盘，Saver可以自动的管理Checkpoints文件。
    例如，我们可以指定保存最近的N个Checkpoints文件。
'''
checkpoint_path = "./checkpoint_dir/yolo2_coco.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)# 载入模型参数
    bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})#运行得到结果

bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                         image_shape=image_shape)# 后处理
img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)# 画出结果框
cv2.imwrite("detection.jpg", img_detection)# 保存检测后的图像
cv2.imshow("detection results", img_detection)# 显示结果

cv2.waitKey(0)#等待按键结束
