# -*- coding: utf-8 -*-
# yolo的weights文件转成caffe的caffemodel文件， 这是模型的参数
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw
"""

import caffe
import numpy as np
import sys, getopt

def main(argv):
	model_filename = ''
	yoloweight_filename = ''
	caffemodel_filename = ''
	try:
		opts, args = getopt.getopt(argv, "hm:w:o:")
		print opts
	except getopt.GetoptError:
		print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			yoloweight_filename = arg
		elif opt == "-o":
			caffemodel_filename = arg
			
	print 'model file is ', model_filename
	print 'weight file is ', yoloweight_filename
	print 'output caffemodel file is ', caffemodel_filename
	net = caffe.Net(model_filename, caffe.TEST)
	params = net.params.keys()

	# read weights from file and assign to the network
	netWeightsInt = np.fromfile(yoloweight_filename, dtype=np.int32)
	transFlag = (netWeightsInt[0]>1000 or netWeightsInt[1]>1000) # transpose flag, the first 4 entries are major, minor, revision and net.seen
	print transFlag

	netWeightsFloat = np.fromfile(yoloweight_filename, dtype=np.float32)
	netWeights = netWeightsFloat[4:] # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
	print netWeights.shape
	count = 0
	for pr in params:
		lidx = list(net._layer_names).index(pr)
		layer = net.layers[lidx]
		if count == netWeights.shape[0] and (layer.type != 'BatchNorm' and layer.type != 'Scale'):
			print "WARNING: no weights left for %s" % pr
			break
		if layer.type == 'Convolution':
			print pr+"(conv)"
			# bias
			if len(net.params[pr]) > 1:
				bias_dim = net.params[pr][1].data.shape
			else:
				bias_dim = (net.params[pr][0].data.shape[0], )
			biasSize = np.prod(bias_dim)
			conv_bias = np.reshape(netWeights[count:count+biasSize], bias_dim)
			if len(net.params[pr]) > 1:
				assert(bias_dim == net.params[pr][1].data.shape)
				net.params[pr][1].data[...] = conv_bias
				conv_bias = None
			count = count + biasSize
			# batch_norm
			next_layer = net.layers[lidx+1]
			if next_layer.type == 'BatchNorm':
				bn_dims = (3, net.params[pr][0].data.shape[0])
				bnSize = np.prod(bn_dims)
				batch_norm = np.reshape(netWeights[count:count+bnSize], bn_dims)
				count = count + bnSize
			# weights
			dims = net.params[pr][0].data.shape
			weightSize = np.prod(dims)
			net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
			count = count + weightSize
		elif layer.type == 'InnerProduct':
			print pr+"(fc)"
			# bias
			biasSize = np.prod(net.params[pr][1].data.shape)
			net.params[pr][1].data[...] = np.reshape(netWeights[count:count+biasSize], net.params[pr][1].data.shape)
			count = count + biasSize
			# weights
			dims = net.params[pr][0].data.shape
			weightSize = np.prod(dims)
			if transFlag:
				net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], (dims[1], dims[0])).transpose()
			else:
				net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
			count = count + weightSize
		elif layer.type == 'BatchNorm':
			print pr+"(batchnorm)"
			net.params[pr][0].data[...] = batch_norm[1]	# mean
			net.params[pr][1].data[...] = batch_norm[2]	# variance
			net.params[pr][2].data[...] = 1.0	# scale factor
		elif layer.type == 'Scale':
			print pr+"(scale)"
			net.params[pr][0].data[...] = batch_norm[0]	# scale
			batch_norm = None
			if len(net.params[pr]) > 1:
				net.params[pr][1].data[...] = conv_bias	# bias
				conv_bias = None
		else:
			print "WARNING: unsupported layer, "+pr
	if np.prod(netWeights.shape) != count:
		print "ERROR: size mismatch: %d" % count
	net.save(caffemodel_filename)		
		
if __name__=='__main__':	
	main(sys.argv[1:])
