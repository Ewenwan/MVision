#-*- coding:utf-8 -*-
# 测试集测试 结果保持为文件
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
# 可视化
#import matplotlib.pyplot as plt
# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../../'  # caffe主目录下
sys.path.insert(0, caffe_root + 'python')# python接口

import caffe#需要 make pycaffe
import math

#caffe.set_mode_cpu()
GPU_ID = 1 # 0,1,2,3,4,5,6,7...
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

model_def = './darknet_v3/gnet_region_dect_darknet_v2.prototxt'
model_weights = './../darknet_v3/models/yolov2_models_iter_11000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([105, 117, 123])
# create transformer for the input called 'data'
# 数据处理器
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # w*h*c-> c*w*h 通道维度放在前面
transformer.set_mean('data', mu)            # 每通道均值
transformer.set_raw_scale('data', 255)      # [0, 1] 转换到 [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # 图像通道变换 RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          416, 416)  # image size is 227x227  yolov2 是416*416输入

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))

# 重叠区域
def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left
#交并比
def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area
#非极大值抑制
def apply_nms(boxes, thres):
    # 按得分排序 
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()#记录剔除的框
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        truth =  sorted_boxes[i]# 最高预测值的边框
        for j in range(i+1, len(sorted_boxes)):#之后的边框
            if j in p:
                continue
            box = sorted_boxes[j]#之后的边框参数
            iou = cal_iou(box, truth)#和最高预测值边框 的重叠度
            if iou >= thres:#大于阈值
                p[j] = 1#记录剔除的框
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:#保留留下来的框
            res.append(sorted_boxes[i])
    return res

#  对模型输出结果进行解码 
def parse_result(reg_out):
	# 输出数据维度 13*13个格子 预测5个边框 每个边框 5+20个参数
	swap = np.zeros((13*13,5,25))
	#变换网络输出 change
	for h in range(13):
			for w in range(13):
				for c in range(125):
					swap[h*13+w][c/25][c%25]  = reg_out[c][h][w]
	#预设 格子尺寸
	biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
	boxes = list()# 格子列表
	#boxes = np.zeros((13*13*5, 8), dtype=np.float32)
	for h in range(13):
		for w in range(13):
			for n in range(5):#每个预测边框
				box = list();#边框
				# box = np.zeros((1, 8), dtype=np.float32)
				cls = list();#20个类别物体的预测概率
				s = 0;
				x = (w + sigmoid(swap[h*13+w][n][0])) / 13.0;
				y = (h + sigmoid(swap[h*13+w][n][1])) / 13.0;
				ww = (math.exp(swap[h*13+w][n][2])*biases[2*n]) / 13.0;
				hh = (math.exp(swap[h*13+w][n][3])*biases[2*n+1]) / 13.0;
				obj_score = sigmoid(swap[h*13+w][n][4]);# 目标得分
				for p in range(20):#20类物体的预测
					cls.append(swap[h*13+w][n][5+p]);
				large = max(cls);#最大的预测概率
				for i in range(len(cls)):
					cls[i] = math.exp(cls[i] - large);#减去最大的预测概率之后小于0了
					#再做一个指数映射到0~1之间
				s = sum(cls);#概率之和
				for i in range(len(cls)):
					cls[i] = cls[i] * 1.0 / s;#概率归一化
				box.append(x);#box[0]
				box.append(y);#box[1]
				box.append(ww);#box[2]
				box.append(hh);#box[3]
				box.append(cls.index(max(cls))+1)#最大的类别预测概率id 对应分类物体#box[4]
				box.append(obj_score);#背景/前景概率#box[5]
				box.append(max(cls));#最大的类别预测概率#box[6]
				box.append(obj_score * max(cls))#预测得分#box[7]
				if box[7] > 0.01:#保留前景物体的预测
					boxes.append(box);
	return apply_nms(boxes, 0.7)

def write_boxes(image, boxes, thresh=0.5):
	label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
	ori_w = image.shape[1]
	ori_h = image.shape[0]
	res_name = "./results/yolo2_det_test_";
	if len(boxes)>0:
		for box in boxes:
			if box[7] < thresh:# 0,1,2,3为坐标 4为得分 得分小于阈值的跳过
				continue
			# 每张图片的预测结果
			name = res_name + label_name[box[4]]
			#print name
			fid = open(name+".txt", 'a')#追加
			# 写入内容
			fid.write(image_id[:-4])#图片名
			fid.write(' ')
			fid.write(str(box[7]))#预测得分
			fid.write(' ')
			# 左上点坐标
			xmin = max(0, int((box[0] - box[2] / 2.) * ori_w))
			ymin = max(0, int((box[1] - box[3] / 2.) * ori_h))
			# 右下点坐标
			xmax = min(ori_w - 1, int((box[0] + box[2] / 2.) * ori_w))
			ymax = min(ori_h - 1, int((box[1] + box[3] / 2.) * ori_h))
			# 在文件中写入 预测边框
			fid.write(str(xmin))
			fid.write(' ')
			fid.write(str(ymin))
			fid.write(' ')
			fid.write(str(xmax))
			fid.write(' ')
			fid.write(str(ymax))
			fid.write('\n')
			# 关闭文件
			fid.close()

def det(image, image_id):
	# 数据处理器进行处理
	transformed_image = transformer.preprocess('data', image)
	#plt.imshow(image)
	# 更新网络输入数据
	net.blobs['data'].data[...] = transformed_image
	# 网络前向传播
	### perform classification
	output = net.forward()
	# 一批图像中第一个图像的网络输出
	result = output['conv_reg'][0]  # the output probability vector for the first image in the batch
	# 掉用模型输出解码函数得到边框
	boxes = parse_result(result)
	# 显示边框和类别
	write_boxes(image, boxes, 0.2)

data_root = '../../../data/yolo/';
index = 0;
for line in open('test_2007.txt', 'r'):
	index += 1
	print index
	image_name = line.split(' ')[0]
	image_id = image_name.split('/')[-1]
	# 这里caffe读入的图像会转变到 0~1之间
	image = caffe.io.load_image(data_root + image_name)
	det(image, image_id)
