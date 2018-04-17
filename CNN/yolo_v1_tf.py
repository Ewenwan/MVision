#-*- coding: utf-8 -*-
# 论文https://pjreddie.com/media/files/papers/yolo.pdf
"""
Yolo V1 by tensorflow
"""

import numpy as np
import tensorflow as tf
import cv2

# Leaky ReLU激活函数
def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

class Yolo(object):
    def __init__(self, weights_file, verbose=True):#打印调试信息标志verbose
        self.verbose = verbose
        # detection params
        self.S = 7  # cell size       分割的格子尺寸 7*7个格子
        self.B = 2  # boxes_per_cell  每个格子预测 的 边框数量
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]# 分类的物体类别
        self.C = len(self.classes) # number of classes 类别总数
        
        # offset for box center (top left point of each cell) 格子框的 左上角
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                              [self.B, self.S, self.S]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.threshold = 0.2  # 置信度*类别概率 得分预置 confidence scores threhold  最终显示框
        self.iou_threshold = 0.4 # 交并比 预置
        #  the maximum number of boxes to be selected by non max suppression
        self.max_output_size = 10# 非极大值抑制 选择的框的数量

        self.sess = tf.Session()
        self._build_net()# 创建网络结构
        self._build_detector()# 对网络输出 转换成识别结果
        self._load_weights(weights_file)
    # In 输入： 448*448*3  ， OUT 输出： N*1*1470
    def _build_net(self):
        """build the network"""
        if self.verbose:
            print("Start to build the network ...")
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])# 448*448*3 rgb三通道图像
        net = self._conv_layer(self.images, 1, 64, 7, 2)#N*448*448*3 -> 7*7*3*64/2 -> N*224*224*64
        net = self._maxpool_layer(net, 1, 2, 2)  # 2*2 池化核 步长2            -> 112*112*64
        net = self._conv_layer(net, 2, 192, 3, 1)# 112*112*64 -> 3*3*64*192/1 -> 112*112*192
        net = self._maxpool_layer(net, 2, 2, 2)  # 2*2 池化核 步长2            -> 56*56*192
        net = self._conv_layer(net, 3, 128, 1, 1)# 56*56*192 -> 1*1*192*128/1 -> 56*56*128
        net = self._conv_layer(net, 4, 256, 3, 1)# 56*56*128 -> 3*3*128*256/1 -> 56*56*256
        net = self._conv_layer(net, 5, 256, 1, 1)# 56*56*256 -> 1*1*256*256/1 -> 56*56*256
        net = self._conv_layer(net, 6, 512, 3, 1)# 56*56*256 -> 3*3*256*512/1 -> 56*56*512
        net = self._maxpool_layer(net, 6, 2, 2)  # 2*2 池化核 步长2            -> 28*28*512
        net = self._conv_layer(net, 7, 256, 1, 1)# 28*28*512 -> 1*1*512*256/1 -> 28*28*256
        net = self._conv_layer(net, 8, 512, 3, 1)# 28*28*256 -> 3*3*256*512/1 -> 28*28*512
        net = self._conv_layer(net, 9, 256, 1, 1)# 28*28*512 -> 1*1*512*256/1 -> 28*28*256
        net = self._conv_layer(net, 10, 512, 3, 1)#28*28*256 -> 3*3*256*512/1 -> 28*28*512
        net = self._conv_layer(net, 11, 256, 1, 1)#28*28*512 -> 1*1*512*256/1 -> 28*28*256
        net = self._conv_layer(net, 12, 512, 3, 1)#28*28*256 -> 3*3*256*512/1 -> 28*28*512
        net = self._conv_layer(net, 13, 256, 1, 1)#28*28*512 -> 1*1*512*256/1 -> 28*28*256
        net = self._conv_layer(net, 14, 512, 3, 1)#28*28*256 -> 3*3*256*512/1 -> 28*28*512
        net = self._conv_layer(net, 15, 512, 1, 1)#28*28*512 -> 1*1*512*512/1 -> 28*28*512
        net = self._conv_layer(net, 16, 1024, 3, 1)#28*28*512-> 3*3*512*1024/1-> 28*28*1024
        net = self._maxpool_layer(net, 16, 2, 2)   #2*2 池化核 步长2            -> 14*14*1024
        net = self._conv_layer(net, 17, 512, 1, 1) #14*14*1024 -> 1*1*1024*512/1 -> 14*14*512
        net = self._conv_layer(net, 18, 1024, 3, 1)#14*14*512  -> 3*3*512*1024/1 -> 14*14*1024
        net = self._conv_layer(net, 19, 512, 1, 1) #14*14*1024 -> 1*1*1024*512/1 -> 14*14*512
        net = self._conv_layer(net, 20, 1024, 3, 1)#14*14*512  -> 3*3*512*1024/1 -> 14*14*1024
        net = self._conv_layer(net, 21, 1024, 3, 1)#14*14*1024 -> 3*3*1024*1024/1-> 14*14*1024
        net = self._conv_layer(net, 22, 1024, 3, 2)#14*14*1024 -> 3*3*1024*1024/2-> 7*7*1024
        net = self._conv_layer(net, 23, 1024, 3, 1)#7*7*1024   -> 3*3*1024*1024/1-> 7*7*1024
        net = self._conv_layer(net, 24, 1024, 3, 1)#7*7*1024   -> 3*3*1024*1024/1-> 7*7*1024
        net = self._flatten(net)# 7*7*1024 -> 1* 7*7*1024
        net = self._fc_layer(net, 25, 512, activation=leak_relu)# 1*512
        net = self._fc_layer(net, 26, 4096, activation=leak_relu)# 1*4096
        net = self._fc_layer(net, 27, self.S*self.S*(self.C+5*self.B))#1* s*s *((4+1)*b+c) = 7*7*(5*2+20) = 1*1470
        # 1470  =  前980 类别预测概率 + 98 边框置信度  + 196 边框1参数 + 196 边框2参数
        self.predicts = net

    def _build_detector(self):
        """Interpret the net output and get the predicted boxes"""
        # the width and height of orignal image
        self.width = tf.placeholder(tf.float32, name="img_w")
        self.height = tf.placeholder(tf.float32, name="img_h")
        # get class prob, confidence, boxes from net output
        idx1 = self.S * self.S * self.C# 总 类别预测数量  7*7*20 = 980
        idx2 = idx1 + self.S * self.S * self.B# 总边框数量 + 总 类别预测数量
        # class prediction  类别预测概率   7*7*2=98
        class_probs = tf.reshape(self.predicts[0, :idx1], [self.S, self.S, self.C])
        # confidence        置信度  0/1 * 交并比
        confs = tf.reshape(self.predicts[0, idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x, y, w, h)  7*7*1*4 + 7*7*1*4 = 196
        boxes = tf.reshape(self.predicts[0, idx2:], [self.S, self.S, self.B, 4])# (x,y,w,h)

        # convert the x, y to the coordinates relative to the top left point of the image
        # the predictions of w, h are the square root
        # multiply the width and height of image
        # 得到真实 矩形框 坐标中心 和 长宽尺寸
        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S * self.width,#x小格子占比
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S * self.height,#y
                          tf.square(boxes[:, :, :, 2]) * self.width,#w 0~1 * 图片尺寸
                          tf.square(boxes[:, :, :, 3]) * self.height], axis=3)#h  0~1 * 图片尺寸

        ## 最终得分 置信度*类别预测概率  class-specific confidence scores [S, S, B, C]
        scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)#增加一维

        scores = tf.reshape(scores, [-1, self.C])  # [S*S*B, C]#98个框 每个框 20个预测得分
        boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]#98个框 每个框 四个 边框参数 坐标中心 和 长宽尺寸

        # find each box class, only select the max score
        box_classes = tf.argmax(scores, axis=1)# 在98个框中找到 20个得分中最高的 类别
        box_class_scores = tf.reduce_max(scores, axis=1)#最高的 得分

        # filter the boxes by the score threshold
        filter_mask = box_class_scores >= self.threshold#大于得分显示阈值的
        scores = tf.boolean_mask(box_class_scores, filter_mask)# 对应最终的得分
        boxes = tf.boolean_mask(boxes, filter_mask)#框的位置
        box_classes = tf.boolean_mask(box_classes, filter_mask)#类别

        # non max suppression (do not distinguish different classes)
        # ref: https://tensorflow.google.cn/api_docs/python/tf/image/non_max_suppression
        # box (x, y, w, h) -> box (x1, y1, x2, y2)  得到边框 上四条边的中心点
        _boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],# x-0.5*w
                           boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
        #非极大值抑制 筛选 剔除 重叠度高的边框
        nms_indices = tf.image.non_max_suppression(_boxes, scores,
                                                   self.max_output_size, self.iou_threshold)
        self.scores = tf.gather(scores, nms_indices)
        self.boxes = tf.gather(boxes, nms_indices)
        self.box_classes = tf.gather(box_classes, nms_indices)
        
    # 卷积层              输入  id   卷积核数量   卷积核尺寸   滑动卷积步长
    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        """Conv layer"""
        in_channels = x.get_shape().as_list()[-1]# 输入通道数量
        # 创建卷积权重                              尺寸*尺寸*通道数 * 卷积核数量
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters,]))# 偏置为 输出 卷积核数量 个
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2# 填充
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)# 在输入层 加上pad扩展边
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
        output = leak_relu(tf.nn.bias_add(conv, bias)) #Leaky ReLU激活函数
        if self.verbose:
            print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s" \
                  % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output
    # 全连接层
    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]# 输入通道数量
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out,]))
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("    Layer %d: type=Fc, num_out=%d, output_shape=%s" \
                  % (id, num_out, str(output.get_shape())))
        return output
    # 最大值池化层
    def _maxpool_layer(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")
        if self.verbose:
            print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s" \
                  % (id, pool_size, stride, str(output.get_shape())))
        return output
    # 平滑
    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])
    # 载入网络参数
    def _load_weights(self, weights_file):
        """Load weights from file"""
        if self.verbose:
            print("Start to load weights from file:%s" % (weights_file))
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file)

    def detect_from_file(self, image_file, imshow=True, deteted_boxes_file="boxes.txt",
                     detected_image_file="detected_image.jpg"):
        """Do detection given a image file"""
        # read image
        image = cv2.imread(image_file)
        img_h, img_w, _ = image.shape
        scores, boxes, box_classes = self._detect_from_image(image)
        predict_boxes = []
        for i in range(len(scores)):
            predict_boxes.append((self.classes[box_classes[i]], boxes[i, 0],
                                boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
        self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)
    #从图像上检测 物体
    def _detect_from_image(self, image):
        """Do detection given a cv image"""
        img_h, img_w, _ = image.shape#图像长宽
        img_resized = cv2.resize(image, (448, 448))#resize到固定尺寸 448*448
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)# 转到 RGB 通道
        img_resized_np = np.asarray(img_RGB)#转换成 数组
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0# 像素值归一化到 0~1 之间
        scores, boxes, box_classes = self.sess.run([self.scores, self.boxes, self.box_classes],
                    feed_dict={self.images: _images, self.width: img_w, self.height: img_h})
        return scores, boxes, box_classes
    # 打印结果 显示框选 的结果
    def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
                     detected_image_file=None):
        """Show the detection boxes"""
        img_cp = image.copy()#赋值原图像 因为要修改 画 矩形在上面
        if deteted_boxes_file:
            f = open(deteted_boxes_file, "w")#写文件
        #  draw boxes
        for i in range(len(results)):
            x = int(results[i][1])#中心点坐标
            y = int(results[i][2])#
            w = int(results[i][3]) // 2# 矩形框宽度
            h = int(results[i][4]) // 2# 矩形框高度
            if self.verbose:#打印调试信息
                print("   class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (results[i][0],
                            x, y, w, h, results[i][-1]))

                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if deteted_boxes_file:
                f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
                        str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(1)
        if detected_image_file:
            cv2.imwrite(detected_image_file, img_cp)
        if deteted_boxes_file:
            f.close()

if __name__ == "__main__":
    yolo_net = Yolo("./weights/YOLO_small.ckpt")
    yolo_net.detect_from_file("./test/car.jpg")
