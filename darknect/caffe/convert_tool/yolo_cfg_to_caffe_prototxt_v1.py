# -*- coding: utf-8 -*-
# yolo的cfg文件 转成 caffe的prototxt文件，这是模型的配置文件，是描述模型的
# 这里一些 学习率参数为加入
# .proto和prototxt的区别：
# 两个都是google protobuff的文件，proto用于定义结构体参数，.prototxt用于初始化proto中相应的结构体。
from ConfigParser import ConfigParser# 配置文件读写
from collections import OrderedDict# 集合模块, 提供了许多有用的集合类
# OrderedDict的Key会按照插入的顺序排列，不是Key本身排序  有序字典 FIFO
import argparse # 命令行参数解析
import logging  # 运行日志
import os
import sys
###################################################################################
# caffe 层描述格式 通用部分 #######################################################
class CaffeLayerGenerator(object):
    def __init__(self, name, ltype):
        self.name = name#名字
        self.bottom = []
        self.top = []
        self.type = ltype#层类型 数据层Data 卷积层Convolution 池化层Pooling
    def get_template(self):
        return """
layer {{{{
  name: "{}"
  type: "{}"
  bottom: "{}"
  top: "{}"{{}}
}}}}""".format(self.name, self.type, self.bottom[0], self.top[0])

#################################################################################
# caffe 输入层 基于 父类 CaffeLayerGenerator#####################################
# 数据层 输入层
class CaffeInputLayer(CaffeLayerGenerator):
    def __init__(self, name, channels, width, height):
        super(CaffeInputLayer, self).__init__(name, 'Input')
        self.channels = channels#通道数量？
        self.width = width# 宽度
        self.height = height# 高度
    def write(self, f):
        f.write("""
input: "{}"
input_shape {{
  dim: 1
  dim: {}
  dim: {}
  dim: {}
}}""".format(self.name, self.channels, self.width, self.height))

################################################################################
# caffe 卷积层  基于 父类 CaffeLayerGenerator###################################
class CaffeConvolutionLayer(CaffeLayerGenerator):
    def __init__(self, name, filters, ksize=None, stride=None, pad=None, bias=True):
        super(CaffeConvolutionLayer, self).__init__(name, 'Convolution')
        self.filters = filters# 卷积核数量  卷积层输出 通道数量
        self.ksize = ksize    # 卷积核尺寸
        self.stride = stride  # 卷积步长
        self.pad = pad        # 填充
        self.bias = bias      # 偏置 
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if not self.bias: opts.append('bias_term: false')
        param_str = """
  convolution_param {{
    num_output: {}{}
  }}""".format(self.filters, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))

##############################################################################		
# caffe 池化层 基于 父类 CaffeLayerGenerator##################################
class CaffePoolingLayer(CaffeLayerGenerator):
    def __init__(self, name, pooltype, ksize=None, stride=None, pad=None, global_pooling=None):
        super(CaffePoolingLayer, self).__init__(name, 'Pooling')
        self.pooltype = pooltype# 池化方法 MAX  AVE
        self.ksize = ksize      # 池化核大小
        self.stride = stride    # 步长
        self.pad = pad          # 填充
        self.global_pooling = global_pooling# 全局 池化
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if self.global_pooling is not None: opts.append('global_pooling: {}'.format('True' if self.global_pooling else 'False'))
        param_str = """
  pooling_param {{
    pool: {}{}
  }}""".format(self.pooltype, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))
		
#############################################################################		
# caffe  全连接层InnerProductLayer  基于 父类 CaffeLayerGenerator############
class CaffeInnerProductLayer(CaffeLayerGenerator):
    def __init__(self, name, num_output):
        super(CaffeInnerProductLayer, self).__init__(name, 'InnerProduct')
        self.num_output = num_output# 输出数量
    def write(self, f):
        param_str = """
  inner_product_param {{
    num_output: {}
  }}""".format(self.num_output)
        f.write(self.get_template().format(param_str))
		
############################################################################
# caffe 批归一化批归一化 BatchNorm  去均值 除以方差#################################
class CaffeBatchNormLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeBatchNormLayer, self).__init__(name, 'BatchNorm')
    def write(self, f):
        param_str = """
  batch_norm_param {
    use_global_stats: true
  }"""
        f.write(self.get_template().format(param_str))
		
###########################################################################
# caffe 尺度变换层 ########################################################
# Scale层主要完成 top=alpha*bottom + beta的过程，
# 输入 bottom
# 输出 top
# 则层中主要有两个参数alpha与beta,
# 求导会比较简单。
# 
class CaffeScaleLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeScaleLayer, self).__init__(name, 'Scale')
    def write(self, f):
        param_str = """
  scale_param {
    bias_term: true
  }"""
        f.write(self.get_template().format(param_str))
		
###########################################################################
# 非线性激活层 y = max(0,x) ###############################################
class CaffeReluLayer(CaffeLayerGenerator):
    def __init__(self, name, negslope=None):
        super(CaffeReluLayer, self).__init__(name, 'Relu')
        self.negslope = negslope
    def write(self, f):
        param_str = ""
        if self.negslope is not None:
            param_str = """
  relu_param {{
    negative_slope: {}
  }}""".format(self.negslope)
        f.write(self.get_template().format(param_str))
		
##########################################################################
# caffe 随机失活层 #######################################################
class CaffeDropoutLayer(CaffeLayerGenerator):
    def __init__(self, name, prob):
        super(CaffeDropoutLayer, self).__init__(name, 'Dropout')
        self.prob = prob# 失活概率
    def write(self, f):
        param_str = """
  dropout_param {{
    dropout_ratio: {}
  }}""".format(self.prob)
        f.write(self.get_template().format(param_str))
		
########################################################################		
# softmax 分类层 指数映射后 归一化#############################################
class CaffeSoftmaxLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeSoftmaxLayer, self).__init__(name, 'Softmax')
    def write(self, f):
        f.write(self.get_template().format(""))

		
########################################################################
########################################################################		
########################################################################	
# caffe 格式文件 Proto #################################################
class CaffeProtoGenerator:
    def __init__(self, name):
        self.name = name   # 文件名 ---> 类名
        self.sections = [] # 层参数记录 列表
        self.lnum = 0      # 层数量记录
        self.layer = None
    def add_layer(self, l):# 添加一层
        self.sections.append( l )# 列表中添加每一层
		
	# 数据输入层 #################################################
    def add_input_layer(self, items):
        self.lnum = 0
        lname = "data"# 数据层 输入层               通道数量 3      图像宽度     图像宽度
        self.layer = CaffeInputLayer(lname, items['channels'], items['width'], items['height'])
        self.layer.top.append( lname )# 本层输出 top
        self.add_layer( self.layer )  # 网络 添加这一层
	# 卷积层 #####################################################
    def add_convolution_layer(self, items):
        self.lnum += 1                 # 层数量+1
        prev_blob = self.layer.top[0]  # 上一层输出 为 这一层的输入
        lname = "conv" + str(self.lnum)# 层名字
        filters = items['filters']     # 卷积核数量
        ksize = items['size'] if 'size' in items else None     # 卷积核尺寸
        stride = items['stride'] if 'stride' in items else None# 滑动步长
        pad = items['pad'] if 'pad' in items else None         # 填充
        bias = not bool(items['batch_normalize']) if 'batch_normalize' in items else True# 有BN层 就 不加偏置
        self.layer = CaffeConvolutionLayer( lname, filters, ksize=ksize, stride=stride, pad=pad, bias=bias )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( lname )       # 本层输出
        self.add_layer( self.layer )         # 网络 添加这一层
	# 全连接层 ####################################################
    def add_innerproduct_layer(self, items):
        self.lnum += 1                 # 层数量+1
        prev_blob = self.layer.top[0]  # 上一层输出 为 这一层的输入
        lname = "fc" + str(self.lnum)  # 层名字
        num_output = items['output']   # 输出数量
        self.layer = CaffeInnerProductLayer( lname, num_output )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( lname )       # 本层输出
        self.add_layer( self.layer )         # 网络 添加这一层
	# 池化层 ######################################################
    def add_pooling_layer(self, ltype, items, global_pooling=None):
	    # 池化层不计入层数量
        prev_blob = self.layer.top[0]  # 上一层输出 为 这一层的输入
        lname = "pool"+str(self.lnum)  # 层名字
        ksize = items['size'] if 'size' in items else None     # 池化核尺寸
        stride = items['stride'] if 'stride' in items else None# 滑动步长
        pad = items['pad'] if 'pad' in items else None         # 填充
        self.layer = CaffePoolingLayer( lname, ltype, ksize=ksize, stride=stride, pad=pad, global_pooling=global_pooling )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( lname )       # 本层输出
        self.add_layer( self.layer )         # 网络 添加这一层
    # 批归一化层 #################################################
    def add_batchnorm_layer(self, items):
	    # 批归一化层不计入层数量
        prev_blob = self.layer.top[0]  # 上一层输出 为 这一层的输入
        lname = "bn"+str(self.lnum)    # 层名字
        self.layer = CaffeBatchNormLayer( lname )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( lname )       # 本层输出
        self.add_layer( self.layer )         # 网络 添加这一层
	# 尺度变换层 #################################################
    def add_scale_layer(self, items):
	    # 尺度变换层不计入层数量
        prev_blob = self.layer.top[0]  # 上一层输出 为 这一层的输入
        lname = "scale"+str(self.lnum) # 层名字
        self.layer = CaffeScaleLayer( lname )
        self.layer.bottom.append( prev_blob )# 本层输入 
        self.layer.top.append( lname )       # 本层输出
        self.add_layer( self.layer )         # 网络 添加这一层
	# 非线性激活层 ###############################################
    def add_relu_layer(self, items):
	    # 非线性激活层不计入层数量
        prev_blob = self.layer.top[0] # 上一层输出 为 这一层的输入
        lname = "relu"+str(self.lnum) # 层名字
        self.layer = CaffeReluLayer( lname )
        self.layer.bottom.append( prev_blob )# 本层输入 
        self.layer.top.append( prev_blob )   # loopback 名字不变
        self.add_layer( self.layer )         # 网络 添加这一层
    # 随机失活层 #################################################
    def add_dropout_layer(self, items):
	    # 随机失活层 不计入层数量
        prev_blob = self.layer.top[0]# 上一层输出 为 这一层的输入
        lname = "drop"+str(self.lnum)# 层名字
        self.layer = CaffeDropoutLayer( lname, items['probability'] )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( prev_blob )   # loopback   名字不变
        self.add_layer( self.layer )         # 网络 添加这一层
	# 分类层 #####################################################
    def add_softmax_layer(self, items):
	    # 分类层 不计入层数量
        prev_blob = self.layer.top[0]# 上一层输出 为 这一层的输入
        lname = "prob"               # 层名字 
        self.layer = CaffeSoftmaxLayer( lname )
        self.layer.bottom.append( prev_blob )# 本层输入
        self.layer.top.append( lname )       # 本层输出 
        self.add_layer( self.layer )         # 网络 添加这一层
	# 设置输出  #################################################
    def finalize(self, name):
        self.layer.top[0] = name    # replace
	# 生成cafe prototxt文件 #####################################
    def write(self, fname):
        with open(fname, 'w') as f:
            f.write('name: "{}"'.format(self.name))
            for sec in self.sections:
                sec.write(f)
        logging.info('{} is generated'.format(fname))

#################################################################
# 字典 关键字 不可重复
class uniqdict(OrderedDict):
    _unique = 0
    def __setitem__(self, key, val):
        if isinstance(val, OrderedDict):
            self._unique += 1
            key += "_" + str(self._unique)
        OrderedDict.__setitem__(self, key, val)

### darknet cfg 转 caffe的prototxt文件  
def convert(cfgfile, ptxtfile):
    #
    parser = ConfigParser(dict_type=uniqdict)# 传入一个 独一无二的字典
    parser.read(cfgfile)# 读入  darknet cfg
    netname = os.path.basename(cfgfile).split('.')[0]# 文件基础名字
    print netname
    gen = CaffeProtoGenerator(netname)#创建一个类
    for section in parser.sections():
        _section = section.split('_')[0]
        if _section in ["crop", "cost"]:
            continue
        #
        batchnorm_followed = False
        relu_followed = False
        items = dict(parser.items(section))
		# BN 
        if 'batch_normalize' in items and items['batch_normalize']:
            batchnorm_followed = True
	    # 激活 
        if 'activation' in items and items['activation'] != 'linear':
            relu_followed = True
        # [net] 标签 数据层  height=448 width=448 channels=3
        if _section == 'net':
            gen.add_input_layer(items)
	    # [convolutional] 标签 圈基层
        elif _section == 'convolutional':
            gen.add_convolution_layer(items)
            if batchnorm_followed:       # 添加 BN层
                gen.add_batchnorm_layer(items)
                gen.add_scale_layer(items)
            if relu_followed:
                gen.add_relu_layer(items)# 添加激活层
		# [connected] 标签 全连接层
        elif _section == 'connected':
            gen.add_innerproduct_layer(items)
            if relu_followed:
                gen.add_relu_layer(items)# 添加激活层
		# [maxpool] 标签 最大值池化层
        elif _section == 'maxpool':
            gen.add_pooling_layer('MAX', items)
	    # [avgpool] 标签 均值池化层
        elif _section == 'avgpool':
            gen.add_pooling_layer('AVE', items, global_pooling=True)
	    # [dropout] 标签 随机失活层
        elif _section == 'dropout':
            gen.add_dropout_layer(items)
		#  [dropout] 标签 分类 
        elif _section == 'softmax':
            gen.add_softmax_layer(items)
        else:
            logging.error("{} layer is not supported".format(_section))
    #gen.finalize('result')
    gen.write(ptxtfile)

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO cfg to Caffe prototxt')
    parser.add_argument('cfg', type=str, help='YOLO cfg')
    parser.add_argument('prototxt', type=str, help='Caffe prototxt')
    args = parser.parse_args()

    convert(args.cfg, args.prototxt)

if __name__ == "__main__":
    main()

# vim:sw=4:ts=4:et
