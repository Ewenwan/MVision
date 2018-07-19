#-*- coding:utf-8 -*-
import numpy as np
import cv2
import os, sys
# export PYTHONPATH=/home/wanyouwen/ewenwan/software/caffe_yolo/caffe-yolov2_oldVer/python
caffe_root = '../../../'  # caffe主目录下
sys.path.insert(0, caffe_root + 'python')# python接口
import caffe#需要 make pycaffe
import math

#caffe.set_mode_cpu()
GPU_ID = 1 # 0,1,2,3,4,5,6,7...
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))

#重合部分的边长
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
	#index = 0
	for h in range(13):#行格子
			for w in range(13):#列数
				for c in range(125):#框数量
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
				# 边框参数
				box.append(x);#box[0]
				box.append(y);#box[1]
				box.append(ww);#box[2]
				box.append(hh);#box[3]
				#最大类别概率 id
				#box.append(np.argsort(cls)[-1])
				box.append(cls.index(max(cls))+1)#最大的类别预测概率id 对应分类物体#box[4]
				box.append(obj_score);#背景/前景概率#box[5]
				box.append(max(cls));#最大的类别预测概率#box[6]
				box.append(obj_score * max(cls))#预测得分#box[7]
				#print box[7]
				if box[7] > 0.01:#保留前景物体的预测
					boxes.append(box);
	print "boxs before nms: "
	print len(boxes)
	return apply_nms(boxes, 0.5)
	
def show_boxes(im_path, boxes, thresh=0.5, show=0):
	import random
	r = lambda: random.randint(0,255)
	print "dect: "
	print len(boxes)
	# voc 数据集 20类
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
	im = cv2.imread(im_path)
	# 原图像尺寸
	ori_w = im.shape[1]
	ori_h = im.shape[0]
	# 对于每一个预测边框
	if len(boxes)>0:
		for box in boxes:
			if box[7] < thresh:# 0,1,2,3为坐标 7为得分 得分小于阈值的跳过
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
			cv2.putText(im, classes[int(box[4])-1] + " : %.2f" % box[7],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
	else:
		print "No Object."
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
	im = cv2.resize(im, (416, 416))#宽，高，通道
	# 通道 宽高
	im = np.require(im.transpose((2, 0, 1)), dtype=np.float32)
	# 去均值
	im -= mean
	# 更新网络输入数据
	model.blobs['data'].data[...] = im
	# 网络前向传播
	output_blobs = model.forward()
	# 一批图像中第一个图像的网络输出
	result = output_blobs['conv_reg'][0]
	# 掉用模型输出解码函数得到边框
	boxes = parse_result(result)
	# 显示边框和类别
	show_boxes(im_path, boxes, 0.8)

if __name__=="__main__":
	if len(sys.argv) > 1:
		im_path = sys.argv[1]
	else:
		im_path = './dog.jpg'
	#else:
	#im_path = './dog.jpg'
	model_def = './darknet_v3/my_yolo_v2_test.prototxt'
	#model_weights = './../darknet_v3/models/yolov2_models_iter_15000.caffemodel'
	model_weights = './../darknet_v3/models/yolov2_models_iter_32000.caffemodel'
	model = caffe.Net(model_def,      # 模型文件
					  model_weights,  # 权重文件
					  caffe.TEST)     # 测试模型 (e.g., don't perform dropout)
	det(model, im_path)#显示图像
