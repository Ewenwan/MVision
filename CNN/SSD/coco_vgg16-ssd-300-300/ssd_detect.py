#-*- coding:utf-8 -*-
# ssd_300_300_coco数据集
'''
Detection with SSD
In this example, we will load a SSD model 
and use it to detect objects.
'''
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
#caffe_root = '../../../'  # caffe主目录下
#os.chdir(caffe_root)
#sys.path.insert(0, os.path.join(caffe_root, 'python'))
# 可以事先导入caffe路径
#sudo echo export PYTHONPATH="/home/wanyouwen/ewenwan/software/caffe-ssd/python" >> ~/.bashrc
#source ~/.bashrc
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# 由标签 映射map 根据整数标签得到字符串标签
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)#标签数量
    labelnames = []#标签对应得名字
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# caffe 目标检测器
class CaffeDetection:
    # 类初始化函数     gpuid    模型文件   权重文件        图片大小      标签map
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)#gpu编号
        caffe.set_mode_gpu()#gpu模式

        self.image_resize = image_resize
        # 定义网络
        self.net = caffe.Net(model_def,# 网络模型
                             model_weights,# 网络权重
                             caffe.TEST)# 测试模式 不使用 droupout

        # 输入图片处理变换器: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))# 通道*宽*高
        self.transformer.set_mean('data', np.array([104, 117, 123])) # voc数据集 各个通道均值 mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)#[0,1]转换成[0,255]
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))# RGB --->BGR

        # 图片标签文件 load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)
        
    # 检测器
    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300 # 1张图片 3通道 300*300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)
        # Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # 模型前向传播 并且获取 detection_out 层的输出
        detections = self.net.forward()['detection_out']

        # 模型输出 解码
        det_label = detections[0,0,:,1]# 标签索引
        det_conf = detections[0,0,:,2] # 可信度
        det_xmin = detections[0,0,:,3] # 坐标
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # 获取可行度大于 0.5的索引
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_conf = det_conf[top_indices]# 可信度
        top_label_indices = det_label[top_indices].tolist()# 标签索引
        top_labels = get_labelname(self.labelmap, top_label_indices)# 标签字符串
        top_xmin = det_xmin[top_indices]# 坐标 0~1小数
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        
        # 前5个
        result = []
        for i in xrange(min(topn, top_conf.shape[0])):# 前5个
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]# 预测得分
            label = int(top_label_indices[i])#标签id
            label_name = top_labels[i]#标签字符串
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
            # result[i][0] xmin
            # result[i][1] ymin
            # result[i][2] xmax
            # result[i][3] ymax
            # result[i][4] label
            # result[i][5] score
            # result[i][6] label_name         
        return result
        
def main(args):
    '''main '''
    # 定义一个检测器类
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    # 检测并获取结果
    result = detection.detect(args.image_file)
    # 打印结果
    print result
    #结果显示
    img = Image.open(args.image_file)#打开图像
    draw = ImageDraw.Draw(img)#显示
    width, height = img.size#原来图像大小
    print width, height
    for item in result:
        # 获取坐标实际整数值
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))#红色框
        #                      [6] label_name  [5] score   
        draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))#显示文本标签 绿色
        print item
        print [xmin, ymin, xmax, ymax]
        print [xmin, ymin], item[-1]
    img.save('detect_result.jpg')

# 命令行参数解析
def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=7, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='./labelmap_coco.prototxt')
    parser.add_argument('--model_def',
                        default='./VGG_coco_SSD_300x300_deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='models/VGG_coco_SSD_300x300_iter_9000.caffemodel')
    #parser.add_argument('--image_file', default='../SSD_300x300/fish-bike.jpg')
    #parser.add_argument('--image_file', default='../SSD_300x300/dog.jpg')
    parser.add_argument('--image_file', default='../SSD_300x300/person.jpg')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
    
