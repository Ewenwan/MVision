# -*- coding: utf-8 -*-
# yolo的weights文件转成caffe的caffemodel文件， 这是模型的参数
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw
"""

import caffe
import numpy as np
import sys, getopt
# getopt 用来分析命令行参数。参数argc和argv分别代表参数个数和内容

def main(argv):
    # 命令行读写文件 
	model_filename = ''# caffe prototxt 模型文件
	yoloweight_filename = ''# yolo.weights 模型参数文件
	caffemodel_filename = ''# yolo_caffe   模型参数文件 caffemodel 
	try:
		opts, args = getopt.getopt(argv, "hm:w:o:")# argv[1:] model weights output
		print opts# opts为分析出的格式信息  args 为不属于格式信息的剩余的命令行参数
		# opts 是一个两元组的列表。每个元素为：( 选项串, 附加参数) 。如果没有附加参数则为空串''
	except getopt.GetoptError:
		print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':# help
			print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
			sys.exit()
		elif opt == "-m":# -m 选项
			model_filename = arg
		elif opt == "-w":# -w 选项
			yoloweight_filename = arg
		elif opt == "-o":# -o 选项
			caffemodel_filename = arg
			
	print 'model file is ', model_filename
	print 'weight file is ', yoloweight_filename
	print 'output caffemodel file is ', caffemodel_filename
	
	# 读入 caffe prototxt 模型文件
	net = caffe.Net(model_filename, caffe.TEST)# 测试 方式 读入  caffe prototxt 模型文件
	params = net.params.keys()

	# 读入 yolo.weights 模型参数文件
	netWeightsInt = np.fromfile(yoloweight_filename, dtype=np.int32)
	transFlag = (netWeightsInt[0]>1000 or netWeightsInt[1]>1000) # transpose flag, the first 4 entries are major, minor, revision and net.seen
	print transFlag

	netWeightsFloat = np.fromfile(yoloweight_filename, dtype=np.float32)
	netWeights = netWeightsFloat[4:] # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
	print netWeights.shape
	
	
	count = 0
	for pr in params:# 每一个层 
		lidx = list(net._layer_names).index(pr)
		layer = net.layers[lidx]
		if count == netWeights.shape[0] and (layer.type != 'BatchNorm' and layer.type != 'Scale'):
			print "WARNING: no weights left for %s" % pr
			break
		# 卷积层
		if layer.type == 'Convolution':
			print pr + "(conv)"
			# 偏置 bias
			if len(net.params[pr]) > 1:
				bias_dim = net.params[pr][1].data.shape
			else:
				bias_dim = (net.params[pr][0].data.shape[0], )
			biasSize = np.prod(bias_dim)
			# 卷积权重
			conv_bias = np.reshape(netWeights[count:count+biasSize], bias_dim)
			if len(net.params[pr]) > 1:
				assert(bias_dim == net.params[pr][1].data.shape)
				net.params[pr][1].data[...] = conv_bias
				conv_bias = None
			count = count + biasSize
			# BN层 batch_norm
			next_layer = net.layers[lidx+1]
			if next_layer.type == 'BatchNorm':
				bn_dims = (3, net.params[pr][0].data.shape[0])
				bnSize = np.prod(bn_dims)
				batch_norm = np.reshape(netWeights[count:count+bnSize], bn_dims)
				count = count + bnSize
			# 权重weights
			dims = net.params[pr][0].data.shape
			weightSize = np.prod(dims)
			net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
			count = count + weightSize
		# 全链接层
		elif layer.type == 'InnerProduct':
			print pr+"(fc)"
			# bias 偏置
			biasSize = np.prod(net.params[pr][1].data.shape)
			net.params[pr][1].data[...] = np.reshape(netWeights[count:count+biasSize], net.params[pr][1].data.shape)
			count = count + biasSize
			# weights 权重
			dims = net.params[pr][0].data.shape
			weightSize = np.prod(dims)
			if transFlag:
				net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], (dims[1], dims[0])).transpose()
			else:
				net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
			count = count + weightSize
		# BN层
		elif layer.type == 'BatchNorm':
			print pr+"(batchnorm)"
			net.params[pr][0].data[...] = batch_norm[1]	# mean     均值
			net.params[pr][1].data[...] = batch_norm[2]	# variance 方差
			net.params[pr][2].data[...] = 1.0	# scale factor     缩放
		elif layer.type == 'Scale':
			print pr+"(scale)"
			net.params[pr][0].data[...] = batch_norm[0]	# scale    缩放
			batch_norm = None
			if len(net.params[pr]) > 1:
				net.params[pr][1].data[...] = conv_bias	# bias     平移
				conv_bias = None
		else:
			print "WARNING: unsupported layer, "+pr
	if np.prod(netWeights.shape) != count:
		print "ERROR: size mismatch: %d" % count
	net.save(caffemodel_filename)		
		
if __name__=='__main__':	
	main(sys.argv[1:])# 从 第二个参数开始
