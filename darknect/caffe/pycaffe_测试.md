
# 测试单层
```
# test_east
input: "_tadd_blob137"
input_shape {
  dim: 1
  dim: 8
  dim: 128
  dim: 128
}
input: "sigmoid_blob138"
input_shape {
  dim: 1
  dim: 1
  dim: 128
  dim: 128
}
layer {
  name: "east_out"
  type: "EastOutput"
  bottom: "_tadd_blob137"
  bottom: "sigmoid_blob138"
  top: "output"
  east_out_param {
    stride: 4
    score_thre: 0.8
    nms_thre: 0.01
    nms_method: 2
  }
}


```
```py

#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import sys,os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

### caffe路径 加入python环境路径 #####
add_path("/data3/plat/wanyouwen/wanyouwen/cvt_bin_demo/tool/transfer_bin")

os.system("export LD_LIBRARY_PATH=/data3/plat/wanyouwen/wanyouwen/cvt_bin_demo/tool/transfer_bin:$LD_LIBRARY_PATH")


import numpy as np
import caffe , cv2
from caffe.proto import caffe_pb2
from caffe import layers as L
import google.protobuf as pb
import google.protobuf.text_format


if __name__ == '__main__':
    # 解析命令行参数
    #if args.cpu_mode:
    caffe.set_mode_cpu()
    # 创建网络
    test_net="test_east_out.prototxt"
    unit_net = caffe.Net(test_net, caffe.TEST)
    print '\nLoaded network {:s}\n'.format(test_net)

    # 读入数据  每行一个数据
    # input1
    #float_data_power = np.loadtxt("layer_id_98_transed_power136_128x128x8x1_out_0_whcn")
    float_data_power = np.loadtxt("wk_transed_power136.txt")
    float_data_power.shape=[1,8,128,128]
    float_data_power = float_data_power.astype(np.float32)
    # input2
    #float_data_sigmoid = np.loadtxt("layer_id_99_transed_sigmoid137_128x128x1x1_out_0_whcn")
    float_data_sigmoid = np.loadtxt("wk_transed_sigmoid137.txt")
    float_data_sigmoid.shape=[1,1,128,128]
    float_data_sigmoid = float_data_sigmoid.astype(np.float32)
    #需要知道 输入blob的名字
    
    forward_kwargs = {"_tadd_blob137" : float_data_power}
    forward_kwargs["sigmoid_blob138"] = float_data_sigmoid
    unit_net.blobs["_tadd_blob137"].reshape(*(float_data_power.shape))
    unit_net.blobs["sigmoid_blob138"].reshape(*(float_data_sigmoid.shape))
    
    print "unit_net.forward "
    blobs_out = unit_net.forward(**forward_kwargs)
    print "unit_net.forward down"
    
    
    # 需要知道输出blob的名字
    # 提取输出
    print "extract output  "
    netopt = unit_net.blobs["output"]
    print " ot shape "
    for oi in range(len(netopt.shape)):
        print netopt.shape[oi]
    
    with open("./output.txt",'ab')as fout:
        # txt文本文件   reshape成 n行 1列
        np.savetxt(fout, netopt.data.reshape(-1, 1), fmt='%f', newline='\n')


```


# 单模型测试
```py
#!/usr/bin/env python
# coding: utf-8
'''
# usage:
#python test_caffemodel_demo.py \
  --prototxt m.prototxt \
  --caffemodel m.caffemodel \
  --bgr_data bgr_data \
  --chanhel_mean_file 127.5, 127.5, 127.5
eg:
python test_caffemodel_demo.py  \
    --prototxt model_squeezenet_east_0.61G_fire11_bn_reduceV5-interp.prototxt \
    --caffemodel Squeezenet_EAST_CNR_toushi_kuozeng_190000_mvave.caffemodel \
    --bgr_data east_out_384_640_bgr

python test_caffemodel_demo.py  
    --prototxt VDPR_mbv1_11_512_pool_e30_0427.prototxt \
    --caffemodel VDPR_mbv1_11_512_pool_e30_0427.caffemodel \
    --bgr_data vdpr_cls_bgr_256x32.raw
    
    
python test_caffemodel_demo.py  \
    --prototxt merge_bn_kpnv7_yolov3-tep.prototxt \
    --caffemodel merge_bn_kpn_v6_yolov3_day_crop.caffemodel \
    --bgr_data adas_rec_960_256_bgr
'''
'''
生成指定层的输出数据
'''
import os.path as osp
import sys,os
import argparse

#### 解析命令行参数
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='generate YOLO V3 layer data demo')
    parser.add_argument('--gpu', dest='gpu_id', 
                        help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', 
                        help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--prototxt', dest='prototxt', 
                        help='caffe prototxt path', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', 
                        help='caffe model path', type=str)
    parser.add_argument('--bgr_data', dest='bgr_data',
                        help='bgr_data', type=str)
    parser.add_argument('--chanhel_mean_file', dest='chanhel_mean_file', 
                        help='net chanhel mean file', type=str)
                        
                        
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.prototxt is None or args.caffemodel is None or args.bgr_data is None:
        parser.print_help()
        sys.exit(1)
     
    return args

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

### caffe路径 加入python环境路径 #####
add_path("/data3/plat/wanyouwen/wanyouwen/cvt_bin_demo/tool/transfer_bin")

os.system("export LD_LIBRARY_PATH=/data3/plat/wanyouwen/wanyouwen/cvt_bin_demo/tool/transfer_bin:$LD_LIBRARY_PATH")


import numpy as np
import caffe , cv2
from caffe.proto import caffe_pb2
from caffe import layers as L
import google.protobuf as pb
import google.protobuf.text_format


"""
参数配置   这里是不是要从外部传递过来  从文件载入通道均值数据
"""
# yolov3通道均值
#PIXEL_MEANS = 127.5 #np.array([[[127.5, 127.5, 127.5]]], dtype=np.float32)
#PIXEL_SCALE = 0.007843
#PIXEL_MEANS = 123
#PIXEL_MEANS = np.array([127.5, 127.5, 127.5])
#PIXEL_SCALE = 0.007843

#PIXEL_MEANS = np.array([123.680000, 116.779999, 103.940002])
#PIXEL_SCALE = 1

PIXEL_MEANS = np.array([128, 128, 128])
#PIXEL_SCALE = 0.017
PIXEL_SCALE = 0.017241379310344827

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def save_feature_picture(data, name, image_name=None, padsize = 1, padval = 1):
    data = data[0]
    #print "data.shape1: ", data.shape
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    #print "padding: ", padding
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #print "data.shape2: ", data.shape
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    #print "data.shape3: ", data.shape, n
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #print "data.shape4: ", data.shape
    # 这里 plt 有问题
    #plt.figure()
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    #plt.show()
    if image_name == None:
        img_path = './data/feature_picture/' 
    else:
        img_path = './data/feature_picture/' + image_name + "/"
        check_file(img_path)
    plt.savefig(img_path + name + ".jpg", dpi = 400, bbox_inches = "tight")

def check_file(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    bgr_data = args.bgr_data

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not exist!!!').format(caffemodel))
    if not os.path.isfile(bgr_data):
        raise IOError(('{:s} not exist!!!').format(bgr_data))
        
    #if args.cpu_mode:
    caffe.set_mode_cpu()
    
    
    src_net = caffe_pb2.NetParameter() # 网络prototxt对象
    # 只读方式打开
    with open(prototxt, 'r') as fin:
        # 解析prototxt文件 保存每一层信息到 src_net
        pb.text_format.Merge(fin.read(), src_net)
    #
    
    input_name = src_net.input
    input_size = len(input_name)
    input_name_0 = input_name[0]
    print "input_size " + str(input_size) +  "  input0_name: " + input_name_0
    # 针对这种形式的输入
    '''
    input_shape{
    dim: 1
    dim: 3
    dim: 384 #416 #540
    dim: 640 #864
    }
    '''
    try:
        if len( src_net.input_shape[0].dim) != 4 :
            print "input shape error input dim not equal to 4"
            
        n = int(src_net.input_shape[0].dim[0])
        c = int(src_net.input_shape[0].dim[1])
        h = int(src_net.input_shape[0].dim[2])
        w = int(src_net.input_shape[0].dim[3])
    except:
        # 这种形式
        '''
        input: "blob0"
        input_dim: 1
        input_dim: 3
        input_dim: 32
        input_dim: 256
        '''
        if len( src_net.input_dim) != 4 :
            print "input shape error input dim not equal to 4"
            
        n = int(src_net.input_dim[0])
        c = int(src_net.input_dim[1])
        h = int(src_net.input_dim[2])
        w = int(src_net.input_dim[3])
    
    print "input shape: " + str(n) + " " + str(c) + " " + str(h) + " " + str(w)
    
    
    im_info_input = False
    if input_name[-1] == "im_info":
        im_info_input = True
        
    
    # 默认网络输出 blob 为 最后一层的top
    out_layer = src_net.layer[len(src_net.layer)-1]
    out_blob_name = out_layer.top[0]
    
    # 创建网络
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\nLoaded network {:s}\n'.format(caffemodel)

    # 读入数据
    # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
    # tofile()将数组中的数据以二进制格式写进文件 
    # tofile()输出的数据不保存数组形状和元素类型等信息 
    data = np.fromfile(bgr_data, dtype=np.uint8)
    # 形状外面传进来
    #data.shape = [1,3,384,640]
    data.shape = [n,c,h,w]
    
    float_data = data.astype(np.float32)
    # 去均值和归一化操作
    # 去均值
    float_data -= PIXEL_MEANS.reshape(1,3,1,1)
    # 缩放尺度
    float_data *= PIXEL_SCALE
    
    #需要知道 输入blob的名字
    
    forward_kwargs = {input_name_0: float_data}
    net.blobs[input_name_0].reshape(*(float_data.shape))
    
    if im_info_input:
        im_info = np.array([[h, w, 1.0]], dtype=np.float32)
        forward_kwargs[input_name[-1]] = im_info
    
    
    print "net.forward "
    blobs_out = net.forward(**forward_kwargs)
    print "net.forward down"
    
    
    # 需要知道输出blob的名字
    # 提取输出
    print "extract output  "
    netopt = net.blobs[out_blob_name]
    print " ot shape "
    for oi in range(len(netopt.shape)):
        print netopt.shape[oi]
    
    # blob 保存标记
    blob_save_flag = {}
    for bn in net.blobs:
        #print type(bn)
        #print bn
        blob_save_flag[bn] = 0
    
    # 打印网络每个blob的形状
    #net_blobs =  [(k,v.data) for k,v in net.blobs.items()]
    #print net_blobs[0][0]  # 名字
    #print net_blobs[0][1]  # 形状 (n,c,h,w)
    #for k,v in net.blobs.items():
    #    #save_feature_picture(v.data, k.replace("/", ""))
    
    #for ib in range(len(net_blobs)):
    #    blob = net_blobs[ib]
    #    print "name: " + str(blob[0]) + " shape: " + str(blob[1].shape)
    #    print type(blob[1])
    '''
    # 寻找 在网络中的那一层 的输入或是输出
    save_dir = './dump_bin/'
    check_file(save_dir)
    for i, layer in enumerate(src_net.layer):
        if  layer.type == "BatchNorm" or layer.type == "Scale":
            continue
        file_name_base = "layer_id_"+str(i)
        bottom    = layer.bottom
        top       = layer.top
        for ib in range(len(bottom)):
            b_blob = net.blobs[bottom[ib]]
            b_shape = b_blob.shape
            print len(b_shape)
            while len(b_shape) < 4:
                b_shape.append(1)
            file_name = file_name_base + "_" + layer.name.replace("/", "_") + "_" + str(b_shape[3]) + "x" + str(b_shape[2]) + "x" + str(b_shape[1]) + "x" + str(b_shape[0]) + "_in_" + str(ib) + "_whcn"
            b_blob.data.tofile(save_dir + file_name)
        for ob in range(len(top)):
            o_blob = net.blobs[top[ob]]
            o_shape = o_blob.shape
            while len(o_shape) < 4:
                o_shape.append(1)
            file_name = file_name_base + "_" + layer.name.replace("/", "_") + "_"  + str(o_shape[3]) + "x" + str(o_shape[2]) + "x" + str(o_shape[1]) + "x" + str(o_shape[0]) + "_out_" + str(ib) + "_whcn"
            o_blob.data.tofile(save_dir + file_name)
    '''
    save_dir = './dump_txt/'
    check_file(save_dir)
    layer_num = len(src_net.layer)
    for i, layer in enumerate(src_net.layer):
        #if  layer.type == "BatchNorm" or layer.type == "Scale":
        #    continue
        print "save: " + str(i) + " layer, res: " + str(layer_num-i-1)
        file_name_base = "layer_id_"+str(i)
        bottom    = layer.bottom
        top       = layer.top
        for ib in range(len(bottom)):
            if not blob_save_flag[bottom[ib]]:
                blob_save_flag[bottom[ib]] = 1
                b_blob = net.blobs[bottom[ib]]
                b_shape = b_blob.shape
                while len(b_shape) < 4:
                    b_shape.append(1)
                file_name = file_name_base + "_" + layer.name.replace("/", "_") + "_" + str(b_shape[3]) + "x" + str(b_shape[2]) + "x" + str(b_shape[1]) + "x" + str(b_shape[0]) + "_in_" + str(ib) + "_whcn"
                #b_blob.data.tofile(save_dir + file_name)
                with open(save_dir + file_name, 'ab')as fout:
                    # txt文本文件   reshape成 n行 1列
                    np.savetxt(fout, b_blob.data.reshape(-1, 1), fmt='%f', newline='\n')
        for ob in range(len(top)):
            if not blob_save_flag[top[ob]]:
                blob_save_flag[top[ob]] = 1
                o_blob = net.blobs[top[ob]]
                o_shape = o_blob.shape
                while len(o_shape) < 4:
                    o_shape.append(1)
                file_name = file_name_base + "_" + layer.name.replace("/", "_") + "_"  + str(o_shape[3]) + "x" + str(o_shape[2]) + "x" + str(o_shape[1]) + "x" + str(o_shape[0]) + "_out_" + str(ib) + "_whcn"
                #o_blob.data.tofile(save_dir + file_name)
                with open(save_dir + file_name, 'ab')as fout:
                    # txt文本文件   reshape成 n行 1列
                    np.savetxt(fout, o_blob.data.reshape(-1, 1), fmt='%f', newline='\n')
    
    #打印网络参数形状
    #print [(k,v[0].data.shape) for k,v in net.params.items()]
        
        
    ''' ocr 解析 
    opdata = netopt.data.reshape(1, -1)

    opdata = netopt.data.tolist()
    max_value = []
    max_id = []
    for ni in range(netopt.shape[0]):
        max_value.append(0)
        max_id.append(0)
        for ci in range(netopt.shape[2]):
            if opdata[ni][0][ci] > max_value[ni]:
                max_value[ni] = opdata[ni][0][ci]
                max_id[ni]    = ci
        
    for ii in range(netopt.shape[0]):
        if max_id[ii] != 96 and ( ii > 0  and max_id[ii] != max_id[ii-1] ):
            print "id : " + str(max_id[ii]) + " conf: " + str(max_value[ii])
    '''

```
