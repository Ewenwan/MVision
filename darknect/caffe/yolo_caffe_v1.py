# 网络输出 解码 
import caffe
GPU_ID = 1 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
# caffe.set_mode_cpu()
from datetime import datetime
import numpy as np
import sys, getopt
import cv2

def interpret_output(output, img_width, img_height):
    # voc 数据集 20类
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
	w_img = img_width# 图像宽
	h_img = img_height#图像高度
	print w_img, h_img
	threshold = 0.2# 得分 阈值  得分 = 前进/背景概率 * 类别概率
	iou_threshold = 0.5#框重叠 交并比 阈值  预测框之间重合度较高的 框被剔除
	num_class = 20# 类别数量
	num_box = 2#每个格子预测的框数量
	grid_size = 7# 格子 7*7
	probs = np.zeros((7,7,2,20))# 前进/背景概率 * 类别概率
	class_probs = np.reshape(output[0:980],(7,7,20))#前面为类别概率
    #	print class_probs
	scales = np.reshape(output[980:1078],(7,7,2))#两个框的 前景/背景 
    #	print scales
	boxes = np.reshape(output[1078:],(7,7,2,4))# 框坐标尺寸
	offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
	
    # YOLO 框的获取######################
	boxes[:,:,:,0] += offset# x 坐标
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))# y坐标
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0#归一化
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
		
	boxes[:,:,:,0] *= w_img
	boxes[:,:,:,1] *= h_img
	boxes[:,:,:,2] *= w_img
	boxes[:,:,:,3] *= h_img

	for i in range(2):# 两个框
		for j in range(20):
		    # 前进/背景概率 * 类别概率
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
			
    # 按照  得分阈值 剔除效果不好的框####################
	filter_mat_probs = np.array(probs>=threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	probs_filtered = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

	argsort = np.array(np.argsort(probs_filtered))[::-1]
	boxes_filtered = boxes_filtered[argsort]
	probs_filtered = probs_filtered[argsort]
	classes_num_filtered = classes_num_filtered[argsort]
	
	# 剔除重合度较高的	预测框##################
	for i in range(len(boxes_filtered)):# 每一个通过 得分阈值 过滤后的框
		if probs_filtered[i] == 0 : continue
		for j in range(i+1,len(boxes_filtered)):
		    # 过滤掉重叠度较高的框
			if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
				probs_filtered[j] = 0.0
	# 最后得分 	
	filter_iou = np.array(probs_filtered>0.0,dtype='bool')
	boxes_filtered = boxes_filtered[filter_iou]
	probs_filtered = probs_filtered[filter_iou]
	classes_num_filtered = classes_num_filtered[filter_iou]

	result = []
	for i in range(len(boxes_filtered)):
		result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

	return result

# 两个边框的 交并比 
def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

# 显示框 
def show_results(img,results, img_width, img_height):
	img_cp = img.copy()
	disp_console = True
	imshow = True
#	if self.filewrite_txt :
#		ftxt = open(self.tofile_txt,'w')
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2
		if disp_console : print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
		xmin = x-w
		xmax = x+w
		ymin = y-h
		ymax = y+h
		if xmin<0:
			xmin = 0
		if ymin<0:
			ymin = 0
		if xmax>img_width:
			xmax = img_width
		if ymax>img_height:
			ymax = img_height
		if  imshow:
			cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
			print xmin, ymin, xmax, ymax
			cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
			cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)	
	if imshow :
		cv2.imshow('YOLO detection',img_cp)
		cv2.waitKey(1000)



def main(argv):
	model_filename = ''  # 模型文件  prototxt     -m
	weight_filename = '' # 权重文件  caffemodel   -w
	img_filename = ''    # 检测的图像             -i
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:")# 获取三个文件
		print opts
	except getopt.GetoptError:
		print 'yolo_main.py -m <model_file> -w <output_file> -i <img_file>'
		sys.exit(2)
	for opt, arg in opts:#每一个参数
		if opt == '-h':# 帮助信息
			print 'yolo_main.py -m <model_file> -w <weight_file> -i <img_file>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg  # 模型文件
		elif opt == "-w":
			weight_filename = arg # 权重文件
		elif opt == "-i":
			img_filename = arg    # 检测的图像
	print 'model file is "', model_filename
	print 'weight file is "', weight_filename
	print 'image file is "', img_filename
	
	# 载入 模型文件 + 权重文件
	net = caffe.Net(model_filename, weight_filename, caffe.TEST)
	# 载入测试图像
	img = caffe.io.load_image(img_filename) # load the image using caffe io
	inputs = img
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	start = datetime.now()
	# 前向传播
	out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
	end = datetime.now()
	elapsedTime = end-start#检测时间
	print 'total time is " milliseconds', elapsedTime.total_seconds()*1000
	print out.iteritems()
	img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	results = interpret_output(out['result'][0], img.shape[1], img.shape[0]) # fc27 instead of fc12 for yolo_small 
	show_results(img_cv,results, img.shape[1], img.shape[0])
	cv2.waitKey(10000)



if __name__=='__main__':	
	main(sys.argv[1:])
	
