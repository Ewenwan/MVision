# coding:utf-8
'''
# demo usage:
# modify_prototxt.py 
--src_prototxt src_net.prototxt \
--dst_prototxt dest_net.prototxt \
--dst_width 1280 --dst_height 720 \
--caffe_path /data/caffe/darwin-caffe-SVN28509_yolov2_ReOrgOri-FeatReshape-shufflenet_eqr_yolov3_190125_ls
'''
import sys
import argparse
import copy
import os.path as osp

'''
本脚本功能：
1.变换 prototxt 模型分辨率 以及 预设区域尺寸 anchors box 参数排布形式
2.吸收BN层 和 Concat_ls层
'''

### 解析命令行参数 ####
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert conv+bn+scale to conv')
    parser.add_argument('--src_prototxt', dest='src_prototxt',
                        help='prototxt file defining the source network',
                        default=None, type=str)
    parser.add_argument('--dst_prototxt', dest='dst_prototxt',
                        help='prototxt file defining the destination network',
                        default=None, type=str)
    parser.add_argument('--dst_width', dest='dst_width',
                        help='width of input image',
                        default=None, type=str)
    parser.add_argument('--dst_height', dest='dst_height',
                        help='height of input image',
                        default=None, type=str)
    parser.add_argument('--caffe_path', dest='caffe_path',
                        help='absolute path of caffe',
                        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.src_prototxt is None or args.dst_prototxt is None or args.dst_width is None or args.dst_height is None:
        parser.print_help()
        sys.exit(1)

    return args

def add_path(path):
    """
    purpose: add path in sys path
    args:
        path: path to be added
    """
    if path not in sys.path:
        sys.path.insert(0,path)
    
### 加入python环境路径 #####
args = parse_args()
# add darwin-caffe to PYTHONPATH 
#caffe_path = osp.join(args.caffe_path,'python')
add_path(args.caffe_path)

#caffe_root = '/data/caffe/darwin-caffe-SVN28509_yolov2_ReOrgOri-FeatReshape-shufflenet_eqr_yolov3_190125_ls/python'
#sys.path.insert(0, caffe_root)
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
import google.protobuf as pb
import google.protobuf.text_format

#'''
def extend_square_brackets(lines, idx, key_str, out_lines):
    if lines[idx].find(key_str) != -1 and \
        (lines[idx].find('#') == -1 or \
        lines[idx].find('#') > lines[idx].find(key_str)) and \
        (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):# 且本行出现 [ 或者下一行出现 [   (是否要加 [在关键字符串之后的条件?)
        #lines[idx].find('_') == -1 and \

        # 找到了关键字符串
        # 且该行 不包括 #
        # 或者该行 的# 在 关键字符串后面出现
        # 且 无 _ 针对 anchors   有待确认  TODO   可以除去这个条件
        
        #anchors_str = ""
        key_str_str = ""
        
        temp_idx = idx
        # 加入从 key_str:[ xxx,xxx, 开始的行 到有 ']' 无# 的行之前
        #while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and lines[temp_idx].find('#') == -1)):
        while temp_idx < len(lines) and \
            (not (lines[temp_idx].find(']') != -1 \
                  and (lines[idx].find('#') == -1 or \
                       lines[temp_idx].find('#') > lines[temp_idx].find(']')))):
            # 在总字符区域查找 
            # 没找到 ]
            # 且 没找大 #  或者  # 出现在 ] 后面
            
            key_str_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
            temp_idx += 1
            
        # 有可能是已经修改过后的文件(就找不到 anchors:[ xxx,xxx,...,xxx], temp_idx 会越界)
        if temp_idx < len(lines):
            idx = temp_idx
            key_str_str += lines[idx] # 加入有 ]行
            key_str_str = ','.join(key_str_str.split())    # Delete redundatn blank
            print key_str_str
            front_idx = key_str_str.find('[') # [ 开始
            tail_idx = key_str_str.find(']')  # ] 结束
            # key_str_str:[ xxx,xxx,...,xxx]
            # 取出 数字部分
            key_str_str = key_str_str[front_idx + 1:tail_idx]
            # 按逗号分隔 出每个数据 生成 列表
            key_str_list = [x for x in key_str_str.split(',') if x != '']
            print key_str_list
            for key in key_str_list:
                out_lines.append("         {}: {}\n".format(key_str,key))
            idx += 1
            #continue  # Skip appending current line to the out_lines
            #这个怎么解决
            
    return idx, out_lines
#'''

# 需要从list 展开的 关键字符串
square_key_str_list = ['anchors', 'clses', 'forward_roi_num_per_cls', 'anchor_scales']

### 变换 prototxt 模型分辨率 以及 预设区域尺寸 anchors box 参数排布形式
def preprocess(src_prototxt, dst_width, dst_height):
    # 只读方式打开
    with open(src_prototxt, 'r') as fin:
        lines = fin.readlines()
    #resolu_dict = {'720': (720, 1280), '1080': (1080, 1920)}
    #if resolution not in resolu_dict:
    #    print("Only support resolution '1080' or '720' ")
    #    exit()
    idx = 0
    out_lines = []
    first_input_layer=True
    while idx < len(lines):
        # 遍历prototxt每一行
        # 1. 修改输入层 input layer 分辨率 参数 ###
        if lines[idx].find('input_shape') != -1 :  
            in_cnt = 0
            out_lines.append(lines[idx])
            idx += 1
            while idx < len(lines) and lines[idx].find('dim') == -1:  # Skip lines until find first dim 
                idx += 1
                continue
            while idx < len(lines) and lines[idx].find('dim') != -1:  # 保存所有有dim的行
                in_cnt += 1
                out_lines.append(lines[idx])
                idx += 1
            if in_cnt == 4 and first_input_layer:  # 只修改第一个 dim=4 的输入层)
                # 修改 输入层分辨率参数 
                out_lines[-2] = "  dim: {}\n".format(dst_height)
                out_lines[-1] = "  dim: {}\n".format(dst_width)
                first_input_layer=False
        '''
        # Find lines with not commented "anchor"
        ### 2. 修改 检测网络 的 预设区域尺寸 anchors box 参数排布形式 ### 
        # 找到 含有anchors的行 并且无#号(未被注释) 也无'_'(排出掉num_anchors:行)
        #if lines[idx].find('anchors') != -1 and lines[idx].find('#') == -1 and lines[idx].find('_') == -1 and (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):
        if lines[idx].find('anchors') != -1 and (lines[idx].find('#') == -1 or lines[idx].find('#') > lines[idx].find('anchors')) and lines[idx].find('_') == -1 and (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):
            anchors_str = ""
            
            temp_idx = idx
            # 加入从 anchors:[ xxx,xxx, 开始的行 到有 ']' 无# 的行之前
            #while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and lines[temp_idx].find('#') == -1)):
            while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and (lines[idx].find('#') == -1 or lines[temp_idx].find('#') > lines[temp_idx].find(']')))):
                anchors_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
                temp_idx += 1
            #while temp_idx < len(lines):
            #    if(lines[temp_idx].find(']') != -1 and lines[temp_idx].find('#') == -1):
            #        break
            #    anchors_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
            #    temp_idx += 1
            #
            #if temp_idx < len(lines):
            #    print lines[temp_idx]
            #print temp_idx
            #print len(lines)
            # 有可能是已经修改过后的文件(就找不到 anchors:[ xxx,xxx,...,xxx])
            if temp_idx < len(lines):
                idx = temp_idx
                anchors_str += lines[idx] # 加入有 ]行
                anchors_str = ','.join(anchors_str.split())    # Delete redundatn blank
                print anchors_str
                front_idx = anchors_str.find('[')
                tail_idx = anchors_str.find(']')
                # anchors:[ xxx,xxx,...,xxx]
                # 取出 数字部分
                anchors_str = anchors_str[front_idx + 1:tail_idx]
                # 按逗号分隔 出每个数据 生成 列表
                anchors_list = [x for x in anchors_str.split(',') if x != '']
                print anchors_list
                for anchor in anchors_list:
                    out_lines.append("         anchors: {}\n".format(anchor))
                idx += 1
                continue  # Skip appending current line to the out_lines
        '''
        
        #idx, out_lines = extend_square_brackets(lines, idx, 'anchors', out_lines)
        '''
        ### 3. 修改 检测网络 的 kpn_proposal_parameter/kpn_output_parameter clses: [xx,xx,xx] 参数排布形式 ### 
        # 找到 含有clses的行 并且无#号(未被注释) 也无'_'(排出掉num_clses:行)
        if lines[idx].find('clses') != -1 and (lines[idx].find('#') == -1 or lines[idx].find('#') > lines[idx].find('clses'))  and lines[idx].find('_') == -1 and (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):
            clses_str = ""
            
            temp_idx = idx
            # 加入从 anchors:[ xxx,xxx, 开始的行 到有 ']' 无# 的行之前
            while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and (lines[idx].find('#') == -1 or lines[temp_idx].find('#') > lines[temp_idx].find(']')) )):
                clses_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
                temp_idx += 1
            
            # 有可能是已经修改过后的文件(就找不到 anchors:[ xxx,xxx,...,xxx])
            if temp_idx < len(lines):
                idx = temp_idx
                clses_str += lines[idx] # 加入有 ]行
                clses_str = ','.join(clses_str.split())    # Delete redundatn blank
                print clses_str
                front_idx = clses_str.find('[')
                tail_idx = clses_str.find(']')
                # clses:[ xxx,xxx,...,xxx]
                # 取出 数字部分
                clses_str = clses_str[front_idx + 1:tail_idx]
                # 按逗号分隔 出每个数据 生成 列表
                clses_list = [x for x in clses_str.split(',') if x != '']
                print clses_list
                for clses in clses_list:
                    out_lines.append("         clses: {}\n".format(clses))
                idx += 1
                continue  # Skip appending current line to the out_lines
        '''
        #idx, out_lines = extend_square_brackets(lines, idx, 'clses', out_lines)
        
        '''
        # 找到 含有forward_roi_num_per_cls的行 并且无#号(未被注释) 也无'_'(排出掉num_clses:行)
        if lines[idx].find('forward_roi_num_per_cls') != -1 and (lines[idx].find('#') == -1 or lines[idx].find('#') > lines[idx].find('forward_roi_num_per_cls'))  and (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):
            clses_str = ""
            
            temp_idx = idx
            # 加入从 anchors:[ xxx,xxx, 开始的行 到有 ']' 无# 的行之前
            while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and (lines[idx].find('#') == -1 or lines[temp_idx].find('#') > lines[temp_idx].find(']')))):
                clses_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
                temp_idx += 1
            
            # 有可能是已经修改过后的文件(就找不到 anchors:[ xxx,xxx,...,xxx])
            if temp_idx < len(lines):
                idx = temp_idx
                clses_str += lines[idx] # 加入有 ]行
                clses_str = ','.join(clses_str.split())    # Delete redundatn blank
                print clses_str
                front_idx = clses_str.find('[')
                tail_idx = clses_str.find(']')
                # clses:[ xxx,xxx,...,xxx]
                # 取出 数字部分
                clses_str = clses_str[front_idx + 1:tail_idx]
                # 按逗号分隔 出每个数据 生成 列表
                clses_list = [x for x in clses_str.split(',') if x != '']
                print clses_list
                for clses in clses_list:
                    out_lines.append("         forward_roi_num_per_cls: {}\n".format(clses))
                idx += 1
                continue  # Skip appending current line to the out_lines
        '''
        #idx, out_lines = extend_square_brackets(lines, idx, 'forward_roi_num_per_cls', out_lines)
        '''
        # 找到 含有anchor_scales的行 并且无#号(未被注释) 也无'_'(排出掉num_clses:行)
        if lines[idx].find('anchor_scales') != -1 and (lines[idx].find('#') == -1 or lines[idx].find('#') > lines[idx].find('anchor_scales'))  and (lines[idx].find('[') != -1 or lines[idx+1].find('[') != -1):
            clses_str = ""
            
            temp_idx = idx
            # 加入从 anchors:[ xxx,xxx, 开始的行 到有 ']' 无# 的行之前
            while temp_idx < len(lines) and (not (lines[temp_idx].find(']') != -1 and (lines[idx].find('#') == -1 or lines[temp_idx].find('#') > lines[temp_idx].find(']')))):
                clses_str += lines[temp_idx].lstrip().strip() # 去左边空格 去左右两边空格
                temp_idx += 1
            
            # 有可能是已经修改过后的文件(就找不到 anchors:[ xxx,xxx,...,xxx])
            if temp_idx < len(lines):
                idx = temp_idx
                clses_str += lines[idx] # 加入有 ]行
                clses_str = ','.join(clses_str.split())    # Delete redundatn blank
                print clses_str
                front_idx = clses_str.find('[')
                tail_idx = clses_str.find(']')
                # clses:[ xxx,xxx,...,xxx]
                # 取出 数字部分
                clses_str = clses_str[front_idx + 1:tail_idx]
                # 按逗号分隔 出每个数据 生成 列表
                clses_list = [x for x in clses_str.split(',') if x != '']
                print clses_list
                for clses in clses_list:
                    out_lines.append("         anchor_scales: {}\n".format(clses))
                idx += 1
                continue  # Skip appending current line to the out_lines
        '''
        #idx, out_lines = extend_square_brackets(lines, idx, 'anchor_scales', out_lines)
        
        for square_key_str in square_key_str_list:
            idx, out_lines = extend_square_brackets(lines, idx, square_key_str, out_lines)
            
        
        ### 4. 加入其它行
        out_lines.append(lines[idx])
        idx += 1
    
    # 5. 可写方式打开 写入修改后的文件
    with open(src_prototxt, 'w')as fout:
        for line in out_lines:
            fout.write(line)

### 转换 prototxt 去除bn层 并 变换到指定分辨率
def process_prototxt(src_prototxt, dst_prototxt, dst_width, dst_height):
    """
    @function: Process original test prototxt for converting bin
    :param src_prototxt:
    :param dst_prototxt:
    :param resolution: Specify image resolution. "720P" [720*1080], "1080P" [1080*1920]
    :return:
    """
    ### 变换 prototxt 模型分辨率 以及 预设区域尺寸 anchors box 参数排布形式 ###
    preprocess(src_prototxt, dst_width, dst_height)  
    src_net = caffe_pb2.NetParameter()
    #  吸收BN层  Scale 被改名为 "findBatchNorm"
    layer_type_set = set([u"BatchNorm", u"BN", u"Scale", u"LBN"])
    
    # conv+BN / conv+LBN / conv+BatchNorm+Scale / conv+Scale
    
    # 只读方式打开
    with open(src_prototxt, 'r') as fin:
        # 解析prototxt文件 保存每一层信息到 src_net
        pb.text_format.Merge(fin.read(), src_net)
        
    ### 吸收 Concat_ls/concat_res层 ###
    concat_loss_name_list=["Concat_ls","concat_res"] # 最后特有的 收集 loss_Px 层的concat层
    
    Concat_ls_layer_exist=False
    # 遍历 网络的每一层
    remove_layer=[]
    # 前向时需要删除的层 Dropout层等
    forward_remove_layer = []
    # 记录非in-place的信息用于修改
    dict_cbn={}       # 记录  BN类层输出 :  卷积层的输出 bottom名字 字典
    dict_activate={}  # 激活层 in-place 修改 前后 名字字典
    
    
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for i, layer in enumerate(src_net.layer):
        '''############ 1. BN 结构处理  ###################'''
        # BN层
        if layer.type in layer_type_set:
            #print(layer.type)
            # BN层的前一层
            pre_layer = src_net.layer[i - 1]
            if pre_layer.type == "Convolution":
                remove_layer.append(layer)
                # 合并BN层到前一层convolution_param参数里面 Set bias of convolution before current layer True
                pre_layer.convolution_param.bias_term = True
                
                # 可能该层在网络最后 有问题
                if ((i + 1) < (len(src_net.layer) - 1)) and (src_net.layer[i + 1].type not in layer_type_set):
                    #### 确保是 卷积+BN  / 卷积+LBN  / 卷积+Scale / 卷积+BatchNorm 的结构
                    dict_cbn[layer.top[0]]=pre_layer.top[0]
            
            #### 确保是 卷积+BatchNorm+Scale
            if i >= 2 and layer.type == "Scale" and pre_layer.type == "BatchNorm" and src_net.layer[i - 2].type == "Convolution":
                remove_layer.append(layer)
                # scale输出的blob名  和 卷积层的输出blob名关联
                dict_cbn[layer.top[0]]=src_net.layer[i - 2].top[0]
                
        '''############ 2. concat+nms 结构处理  ###########'''
        #Concat_ls_layer_exist=False
        # 记录 Concat_ls 层
        if layer.name in concat_loss_name_list:    # Delete concat_res layer
            # 拷贝 名为 Concat_ls 层 的 输入 层列表
            bottom_list = copy.deepcopy(layer.bottom)
            # 拷贝 名为 Concat_ls 层
            tmp_layer = copy.deepcopy(layer)
            Concat_ls_layer_exist=True
        if Concat_ls_layer_exist:
            # 将concat_loss层的输入直接合并到 输入为loss或combine_loss的nms层中
            if layer.name == "nms":
                print layer.bottom 
                if 'loss' == layer.bottom[0]:
                    layer.bottom.remove('loss')
                elif 'combine_loss' == layer.bottom[0]:
                    layer.bottom.remove('combine_loss')
                else:
                    print "error!! only support loss and combine_loss" 
                # 将 concat_loss 层的输入直接合并到 输入为 loss 或 combine_loss的 nms 层中
                layer.bottom.extend(bottom_list)
                
        '''############ 3. Dropout层处理 ##################'''
        if layer.type in ['Dropout']:
            forward_remove_layer.append(layer)
    
    '''# 除去bn 层'''
    for layer in remove_layer:
        print "remove "+layer.name
        src_net.layer.remove(layer)
    '''# 去除 Concat_ls 层'''
    if Concat_ls_layer_exist:
        try:
            #if(tmp_layer):
            src_net.layer.remove(tmp_layer)
        except NameError:
            print("Concat_ls layer not exist")
    '''# 除去 Dropout 层'''
    for layer in forward_remove_layer:
        print "remove "+layer.name
        src_net.layer.remove(layer)
    # 处理 Dropout 层 非 in-place
    for layer in forward_remove_layer:
        # 非 in-place 情况：输入botom != 输出
        if layer.bottom[0] != layer.top[0]:
            print "Dropout layer not in-place "
            for src_layer in src_net.layer:
                for id in range(len(src_layer.bottom)):
                    # 在网络中找到以 Dropout层的输出bottom 为输入的层，也就是该层的后继层
                    if src_layer.bottom[id] == layer.top[0]:
                        #改名字
                        src_layer.bottom[id] = layer.bottom[0]
    
    
    #### 如果 卷积+BatchNorm+Scale的输入输出blob名字不一样，
    #### 那么其后面的层如激活层 Relu的输入blob就需要改名为前面卷积层的输出blob的名字
    #activatre_layer_type=["ReLU","Sigmoid","TanH","Power","PReLU","AbsVal","BNLL","ELU","LeakyReLU"]  #### 处理方式有bug
    #convolution_layer=["Convolution","Deconvolution"]
    #last_layer=None
    #for i, layer in enumerate(src_net.layer):
    #    if layer.type in activatre_layer_type and last_layer.type in convolution_layer:
    #        if len(layer.bottom) != 1 or len(last_layer.top) != 1:
    #            print layer.type + " / " + last_layer.type + " top/bottom numn not 1"
    #        else:
    #            layer.bottom.remove(layer.bottom[0])
    #            layer.bottom.append(last_layer.top[0])
    #    
    #    last_layer=layer
    
    # 修改激活层的 输入名 同时修改激活层为 in-place情况
    activate_layer = ["ReLU","Sigmoid","TanH","Power","PReLU","AbsVal","BNLL","ELU","LeakyReLU"]
    for i, layer in enumerate(src_net.layer):
        if layer.type in activate_layer:
            #print "layername :"+layer.name
            #for id in range(len(layer.bottom)):
            #print layer.bottom[id]
            # dict_cbn: 激活层输出:前层卷积层输出
            # 查找 有已 被删除的 激活层的输出为 输入的 激活层
            if layer.bottom[0] in dict_cbn:
                layer.bottom[0] = dict_cbn[layer.bottom[0]]  # 替换为 对应卷积层的输出
                # 记录 激活层 输出:输入 字典 为处理 激活层的后继层做处理
                dict_activate[layer.top[0]] = layer.bottom[0]
                layer.top[0] = layer.bottom[0]  # 改成 inplace
    '''
    follow_bn_layer = ["Pooling","Convolution"]
    for i, layer in enumerate(src_net.layer):
        if layer.type in follow_bn_layer:
            #print "layername :"+layer.name
            #for id in range(len(layer.bottom)):
            #print layer.bottom[id]
            if layer.bottom[0] in dict_cbn:
                layer.bottom[0] = dict_cbn[layer.bottom[0]]
            
            elif layer.bottom[0] in dict_activate:
                layer.bottom[0] = dict_activate[layer.bottom[0]]
    
    # 处理岔路口合并的情况 + 激活 非 inplace的case
    #merge_layer=["Eltwise","Concat","Crop"]
    merge_layer=["Eltwise","Concat","Crop"]
    for i, layer in enumerate(src_net.layer):
        if layer.type in merge_layer:
            #print "layername :"+layer.name
            for id in range(len(layer.bottom)):
                #print layer.bottom[id]
                if layer.bottom[id] in dict_cbn:
                    layer.bottom[id] = dict_cbn[layer.bottom[id]]
                    
                elif layer.bottom[id] in dict_activate:
                    layer.bottom[id] = dict_activate[layer.bottom[id]]
    '''
    for i, layer in enumerate(src_net.layer):
        for id in range(len(layer.bottom)):
            # BN层 in-place 后，其后继层的调整
            if layer.bottom[id] in dict_cbn:
                layer.bottom[id] = dict_cbn[layer.bottom[id]]
            # 激活层 in-place 后，其后继层的调整
            elif layer.bottom[id] in dict_activate:
                layer.bottom[id] = dict_activate[layer.bottom[id]]
    
    
                
    # 这种处理有bug
    #for dict_cbn_key in dict_cbn:
    #    for layer in src_net.layer:
    #        if layer.type not in activatre_layer_type:
    #            for id in range(len(layer.bottom)):
    #                if layer.bottom[id] == dict_cbn_key:
    #                    layer.bottom[id] = dict_cbn[dict_cbn_key]
    
    
    # 写入修改后的文件
    with open(dst_prototxt, 'w') as fout:
        fout.write(pb.text_format.MessageToString(src_net))


if __name__ == "__main__":
    args = parse_args()
    process_prototxt(args.src_prototxt, args.dst_prototxt, args.dst_width,args.dst_height)
