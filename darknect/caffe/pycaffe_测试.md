

# 测试单层
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


# 
