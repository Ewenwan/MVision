# Caffe-Python接口 数据集生成 训练 分类 检测 剪枝 量化 等
[参考](https://github.com/Ewenwan/Caffe-Python-Tutorial)

## 数据集生成 
generate_lmdb.py
```py
# -*- coding:utf-8 -*-
# 将图像数据生成lmdb数据集
# 1. 生成分类图像数据集
# 2. 生成目标检测图像数据集
import os
import sys
import numpy as np
import random
from caffe.proto import caffe_pb2
from xml.dom.minidom import parse

# 生成分类标签文件
def labelmap(labelmap_file, label_info):
    labelmap = caffe_pb2.LabelMap()
    for i in range(len(label_info)):
        labelmapitem = caffe_pb2.LabelMapItem()
        labelmapitem.name = label_info[i]['name']
        labelmapitem.label = label_info[i]['label']
        labelmapitem.display_name = label_info[i]['display_name']
        labelmap.item.add().MergeFrom(labelmapitem)
    with open(labelmap_file, 'w') as f:
        f.write(str(labelmap))

def rename_img(Img_dir):
    # 重新命名Img,这里假设图像名称表示为000011.jpg、003456.jpg、000000.jpg格式，最高6位，前补0
    # 列出图像，并将图像改为序号名称
    listfile=os.listdir(Img_dir) # 提取图像名称列表
    total_num = 0
    for line in listfile:  #把目录下的文件都赋值给line这个参数
        if line[-4:] == '.jpg':
            newname = '{:0>6}'.format(total_num) +'.jpg'
            os.rename(os.path.join(Img_dir, line), os.path.join(Img_dir, newname))
            total_num+=1         #统计所有图像

def get_img_size():
    pass

def create_annoset(anno_args):
    if anno_args.anno_type == "detection":
        cmd = "E:\Code\windows-ssd/Build/x64/Release/convert_annoset.exe" \
              " --anno_type={}" \
              " --label_type={}" \
              " --label_map_file={}" \
              " --check_label={}" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {} {} {}" \
            .format(anno_args.anno_type, anno_args.label_type, anno_args.label_map_file, anno_args.check_label,
                    anno_args.min_dim, anno_args.max_dim, anno_args.resize_height, anno_args.resize_width, anno_args.backend, anno_args.shuffle,
                    anno_args.check_size, anno_args.encode_type, anno_args.encoded, anno_args.gray, anno_args.root_dir, anno_args.list_file, anno_args.out_dir)
    elif anno_args.anno_type == "classification":
        cmd = "E:\Code\windows-ssd/Build/x64/Release/convert_annoset.exe" \
              " --anno_type={}" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {} {} {}" \
            .format(anno_args.anno_type, anno_args.min_dim, anno_args.max_dim, anno_args.resize_height,
                    anno_args.resize_width, anno_args.backend, anno_args.shuffle, anno_args.check_size, anno_args.encode_type, anno_args.encoded,
                    anno_args.gray, anno_args.root_dir, anno_args.list_file, anno_args.out_dir)
    print cmd
    os.system(cmd)

def detection_list(Img_dir, Ano_dir, Data_dir, test_num):
    # 造成目标检测图像数据库
    # Img_dir表示图像文件夹
    # Ano_dir表示图像标记文件夹，用labelImg生成
    # Data_dir生成的数据库文件地址
    # test_num测试图像的数目
    # 列出图像
    listfile=os.listdir(Img_dir) # 提取图像名称列表

    # 列出图像，并将图像改为序号名称
    total_num = 0
    for line in listfile:  #把目录下的文件都赋值给line这个参数
        if line[-4:] == '.jpg':
            total_num+=1         #统计所有图像

    trainval_num = total_num-test_num # 训练图像数目

    # 生成训练图像及测试图像列表
    test_list_file=open(Data_dir+'/test.txt','w')
    train_list_file=open(Data_dir+'/trainval.txt','w')

    test_list = np.random.randint(0,total_num-1, size=test_num)

    train_list = range(total_num)
    for n in range(test_num):
        train_list.remove(test_list[n])
    random.shuffle(train_list)

    # 测试图像排序，而训练图像不用排序
    test_list = np.sort(test_list)
    # train_list = np.sort(train_list)

    for n in range(trainval_num):
        train_list_file.write(Img_dir + '{:0>6}'.format(train_list[n]) +'.jpg '+ Ano_dir + '{:0>6}'.format(train_list[n]) +'.xml\n')

    for n in range(test_num):
        test_list_file.write(Img_dir + '{:0>6}'.format(test_list[n]) +'.jpg '+ Ano_dir + '{:0>6}'.format(test_list[n]) +'.xml\n')


caffe_root = 'E:/Code/Github/windows_caffe/'
data_root = caffe_root + 'data/mnist/'
Img_dir = data_root + 'JPEGImages/'
Ano_dir = data_root + 'Annotations/'
anno_type = "detection"
test_num = 100

# 第一步，预处理图像，重命名图像名，生成各图像标记信息
# rename_img(Img_dir)
# 然后通过labelImg(可以通过pip install labelImg安装，出现错误可以删除PyQt4的描述）来生成图像的标记

# 第二步，生成分类标签文件
# 编辑label信息
label_info = [
    dict(name='none', label=0, display_name='background'),  # 背景
    dict(name="cat",label=1, display_name='cat'),  # 背景
    dict(name="dog",label=2, display_name='dog'),  # 背景
]
labelmap(data_root+'labelmap_voc.prototxt', label_info)

# 第三步，生成图像及标记的列表文件
if anno_type == "detection":
    detection_list(Img_dir, Ano_dir, data_root, test_num)
else:
    # 分类，生成
    pass

# 第四步，生成lmdb文件
# 初始化信息
anno_args = {}
anno_args['anno_type'] = anno_type
# 仅用于目标检测，lable文件的类型：{xml, json, txt}
anno_args['label_type'] = "xml"
# 仅用于目标检测，label文件地址
anno_args['label_map_file'] = data_root+"labelmap_voc.prototxt"
# 是否检测所有数据有相同的大小.默认False
anno_args['check_size'] = False
# 检测label是否相同的名称，默认False
anno_args['check_label'] = False
# 为0表示图像不用重新调整尺寸
anno_args['min_dim'] = 0
anno_args['max_dim'] = 0
anno_args['resize_height'] = 0
anno_args['resize_width'] = 0
anno_args['backend'] = "lmdb"  # 数据集格式（lmdb, leveldb）
anno_args['shuffle'] = False  # 是否随机打乱图像及对应标签
anno_args['encode_type'] = ""  # 图像编码格式('png','jpg',...)
anno_args['encoded'] = False  # 是否编码，默认False
anno_args['gray'] = False  # 是否视为灰度图，默认False
anno_args['root_dir'] = data_root  # 存放图像文件夹及标签文件夹的根目录
anno_args['list_file'] = data_root + ''  # listfile文件地址
anno_args['out_dir'] = data_root  # 最终lmdb的存在地址

# 生成训练数据集train_lmdb
anno_args['list_file'] = data_root + 'trainval.txt'
create_annoset(anno_args)

# 生成测试数据集train_lmdb
anno_args['list_file'] = data_root + 'test.txt'
create_annoset(anno_args)

```
## 训练 
train_val.py
```py
# -*- coding:utf-8 -*-
# 训练及测试文件
# 训练网络
import caffe
import numpy as np
import matplotlib.pyplot as plt
import math

def crop_network(prune_proto, caffemodel, prune_caffemodel):
    # 截取已知网络的部分层
    #  caffemodel网络权重值并不要求其结构与proto相对应
    # 网络只会取train_proto中定义的结构中权重作为网络的初始权重值
    # 因此，当我们需要截取某些已训练网络的特定层作为新网络的某些层的权重初始值，只需要在其train_proto定义同名的层
    # 之后caffe将在caffemodel中找到与train_proto定义的同名结构，并将其权重作为应用权重初始值。
    # prune_deploy: 选择保留的网络结构层:prototxt
    # caffemodel: 已知网络的权重连接
    # prune_caffemodel：截断网络的权重连接文件
    net = caffe.Net(prune_proto, caffemodel, caffe.TEST)
    net.save(prune_caffemodel)

def train(solver_proto, caffemodel='', is_step=True, savefig=''):
    # 训练模型函数
    # solver_proto: 训练配置文件
    # caffemodel：预设权重值或者快照等，并不要求其结构与网络结构相对应，但只会取与训练网络结构相对应的权重值
    # is_step: True表示按步训练，False表示直接完成训练
    # savefig: 表示要保存的图像训练时损失变化图
    # 设置训练器：随机梯度下降算法
    solver = caffe.SGDSolver(solver_proto)
    if caffemodel!='':
        solver.net.copy_from(caffemodel)

    if is_step==False:
        # 直接完成训练
        solver.solve()
    else:
        # 迭代次数
        max_iter = 10000
        # 每隔100次收集一次数据
        display = 100

        # 每次测试进行100次解算，10000/100
        test_iter = 100
        # 每500次训练进行一次测试（100次解算），60000/64
        test_interval = 500

        # 初始化
        train_loss = np.zeros(int(math.ceil(max_iter * 1.0 / display)))
        test_loss = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))
        test_acc = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))

        # iteration 0，不计入
        solver.step(1)

        # 辅助变量
        _train_loss = 0
        _test_loss = 0
        _accuracy = 0

        # 分步训练
        for it in range(max_iter):
            # 进行一次解算
            solver.step(1)
            # 每迭代一次，训练batch_size张图片
            _train_loss += solver.net.blobs['loss'].data # 最后一层的损失值
            if it % display == 0:
                # 计算平均train loss
                train_loss[int(it / display)] = _train_loss / display
                _train_loss = 0

            # 测试
            if it % test_interval == 0:
                for test_it in range(test_iter):
                    # 进行一次测试
                    solver.test_nets[0].forward()
                    # 计算test loss
                    _test_loss += solver.test_nets[0].blobs['loss'].data
                    # 计算test accuracy
                    _accuracy += solver.test_nets[0].blobs['accuracy'].data
                    # 计算平均test loss
                test_loss[it / test_interval] = _test_loss / test_iter
                # 计算平均test accuracy
                test_acc[it / test_interval] = _accuracy / test_iter
                _test_loss = 0
                _accuracy = 0

                # 绘制train loss、test loss和accuracy曲线
        print '\nplot the train loss and test accuracy\n'
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # train loss -> 绿色
        ax1.plot(display * np.arange(len(train_loss)), train_loss, 'g')
        # test loss -> 黄色
        ax1.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')
        # test accuracy -> 红色
        ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')

        ax1.set_xlabel('iteration')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('accuracy')

        if savefig!='':
            plt.savefig(savefig)
        plt.show()

#CPU或GPU模型转换
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '../../'
# caffe_root = 'E:/Code/Github/windows_caffe/'
model_root = caffe_root + 'models/mnist/'
solver_proto = model_root + 'solver.prototxt'

```

solver.py
```py
# -*- coding:utf-8 -*-
# 生成solver文件
from caffe.proto import caffe_pb2

def solver_file(model_root, model_name):
    s = caffe_pb2.SolverParameter() # 声明solver结构
    s.train_net = model_root+'train.prototxt' # 训练网络结构配置文件
    s.test_net.append(model_root+'test.prototxt') # 测试时网络结构配置文件，测试网络可有多个
    # 每训练迭代test_interval次进行一次测试。
    s.test_interval = 500
    # 每次测试时的批量数，测试里网络可有多个
    s.test_iter.append(100)
    # 最大训练迭代次数
    s.max_iter = 10000
    # 基础学习率
    s.base_lr = 0.01
    # 动量，记忆因子
    s.momentum = 0.9
    # 权重衰减值，遗忘因子
    s.weight_decay = 5e-4
    # 学习率变化策略。可选参数：fixed、step、exp、inv、multistep
    # fixed: 保持base_lr不变；
    # step: 学习率变化规律base_lr * gamma ^ (floor(iter / stepsize))，其中iter表示当前的迭代次数；
    # exp: 学习率变化规律base_lr * gamma ^ iter；
    # inv: 还需要设置一个power，学习率变化规律base_lr * (1 + gamma * iter) ^ (- power)；
    # multistep: 还需要设置一个stepvalue，这个参数和step相似，step是均匀等间隔变化，而multistep则是根据stepvalue值变化；
    #   stepvalue参数说明：
    #       poly: 学习率进行多项式误差，返回base_lr (1 - iter/max_iter) ^ (power)；
    #       sigmoid: 学习率进行sigmod衰减，返回base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))。
    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    s.display = 100 # 每迭代display次显示结果
    s.snapshot = 5000 # 保存临时模型的迭代数
    s.snapshot_prefix = model_root+model_name+'shapshot' # 模型前缀，就是训练好生成model的名字
    s.type = 'SGD' # 训练方法（各类梯度下降法），可选参数：SGD，AdaDelta，AdaGrad，Adam，Nesterov，RMSProp
    s.solver_mode = caffe_pb2.SolverParameter.GPU # 训练及测试模型，GPU或CPU

    solver_file=model_root+'solver.prototxt' # 要保存的solver文件名

    with open(solver_file, 'w') as f:
        f.write(str(s))

caffe_root = '../../'
model_name = 'LeNet5_Mnist_'
# caffe_root = 'E:/Code/Github/windows_caffe/'
model_root = caffe_root + 'models/mnist/'
solver_file(model_root, model_name)

```

## 分类 
classification.py
```py
# -*- coding:utf-8 -*-
# 用于模型的单张图像分类操作
import os
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 分类单张图像img
def classification(img, net, transformer, synset_words):
    im = caffe.io.load_image(img)
    # 导入输入图像
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    start = time.clock()
    # 执行测试
    net.forward()
    end = time.clock()
    print('classification time: %f s' % (end - start))

    # 查看目标检测结果
    labels = np.loadtxt(synset_words, str, delimiter='\t')

    category = net.blobs['prob'].data[0].argmax()

    class_str = labels[int(category)].split(',')
    class_name = class_str[0]
    # text_font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    cv2.putText(im, class_name, (0, im.shape[0]), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 显示结果
    plt.imshow(im, 'brg')
    plt.show()

#CPU或GPU模型转换
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe_root = '../../'
# 网络参数（权重）文件
caffemodel = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
# 网络实施结构配置文件
deploy = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'


img_root = caffe_root + 'data/VOCdevkit/VOC2007/JPEGImages/'
synset_words = caffe_root + 'data/ilsvrc12/synset_words.txt'

# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# 处理图像
while 1:
    img_num = raw_input("Enter Img Number: ")
    if img_num == '': break
    img = img_root + '{:0>6}'.format(img_num) + '.jpg'
    classification(img,net,transformer,synset_words)

```
## 检测 
detection.py
```py
 # -*- coding:utf-8 -*-
# 用于模型的单张图像分类操作
import os
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 分类单张图像img
def detection(img, net, transformer, labels_file):
    im = caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    start = time.clock()
    # 执行测试
    net.forward()
    end = time.clock()
    print('detection time: %f s' % (end - start))

    # 查看目标检测结果
    file = open(labels_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    loc = net.blobs['detection_out'].data[0][0]
    confidence_threshold = 0.5
    for l in range(len(loc)):
        if loc[l][2] >= confidence_threshold:
            xmin = int(loc[l][3] * im.shape[1])
            ymin = int(loc[l][4] * im.shape[0])
            xmax = int(loc[l][5] * im.shape[1])
            ymax = int(loc[l][6] * im.shape[0])
            img = np.zeros((512, 512, 3), np.uint8)  # 生成一个空彩色图像
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)

            # 确定分类类别
            class_name = labelmap.item[int(loc[l][1])].display_name
            # text_font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
            cv2.putText(im, class_name, (xmin, ymax), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 显示结果
    plt.imshow(im, 'brg')
    plt.show()

#CPU或GPU模型转换
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe_root = '../../'
# 网络参数（权重）文件
caffemodel = caffe_root + 'models/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
# 网络实施结构配置文件
deploy = caffe_root + 'models/SSD_300x300/deploy.prototxt'


img_root = caffe_root + 'data/VOCdevkit/VOC2007/JPEGImages/'
labels_file = caffe_root + 'data/VOC0712/labelmap_voc.prototxt'

# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# 处理图像
while 1:
    img_num = raw_input("Enter Img Number: ")
    if img_num == '': break
    img = img_root + '{:0>6}'.format(img_num) + '.jpg'
    detection(img,net,transformer,labels_file)

```
## 剪枝 
prune.py
```py
 # -*- coding:utf-8 -*-
# 用于修剪网络模型
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

# 由稠密变成CSC稀疏矩阵
def dense_to_sparse_csc(W_flatten, num_level):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    csc_W = [] # 存储稀疏矩阵
    csc_indx = []
    indx = 0
    for n in range(len(W_flatten)):
        if W_flatten[n]!=0 or indx == 2**num_level:
            csc_W.append(W_flatten[n])
            csc_indx.append(indx)
            indx = 0
        else:
            indx += 1
    if indx!=0:
        csc_W.append(0.0)
        csc_indx.append(indx-1)
    return np.array(csc_W, dtype=np.float32),np.array(csc_indx, dtype=np.int8)

# 由稠密变成CSC稀疏矩阵
def sparse_to_dense_csc(csc_W, csc_W_indx):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    W_flatten = [] # 存储稠密矩阵
    indx = 0
    for n in range(len(csc_W)):
        if csc_W_indx[n]!=0:
            W_flatten.extend([0]*(csc_W_indx[n]))
        W_flatten.append(csc_W[n])
    return np.array(W_flatten, dtype=np.float32)


def read_sparse_net(filename, net, layers):
    pass

def write_sparse_net(filename, net):
    pass

# 画出各层参数的直方图
def draw_hist_weight(net, layers):
    plt.figure()  # 画图
    layer_num = len(layers)
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data

        plt.subplot(layer_num/2, 2, i)
        numBins = 2 ^ 5
        plt.hist(W.flatten(), numBins, color='blue', alpha=0.8)
        plt.title(layer)
        plt.show()

# 网络模型的参数
def analyze_param(net, layers):

    print '\n=============analyze_param start==============='
    total_nonzero = 0
    total_allparam = 0
    percentage_list = []
    for i, layer in enumerate(layers):
        i += 1
        W = net.params[layer][0].data
        b = net.params[layer][1].data

        print 'W(%s) range = [%f, %f]' % (layer, min(W.flatten()), max(W.flatten()))
        print 'W(%s) mean = %f, std = %f' % (layer, np.mean(W.flatten()), np.std(W.flatten()))
        non_zero = (np.count_nonzero(W.flatten()) + np.count_nonzero(b.flatten())) # 参数非零值
        all_param = (np.prod(W.shape) + np.prod(b.shape)) # 所有参数的数目
        this_layer_percentage = non_zero / float(all_param) # 参数比例
        total_nonzero += non_zero
        total_allparam += all_param
        print 'non-zero W and b cnt = %d' % non_zero
        print 'total W and b cnt = %d' % all_param
        print 'percentage = %f\n' % (this_layer_percentage)
        percentage_list.append(this_layer_percentage)

    print '=====> summary:'
    print 'non-zero W and b cnt = %d' % total_nonzero
    print 'total W and b cnt = %d' % total_allparam
    print 'percentage = %f' % (total_nonzero / float(total_allparam))
    print '=============analyze_param ends ==============='
    return (total_nonzero / float(total_allparam), percentage_list)

def prune(threshold, test_net, layers):
    sqarse_net = {}

    for i, layer in enumerate(layers):

        print '\n============  Pruning %s : threshold=%0.2f   ============' % (layer,threshold[i])
        W = test_net.params[layer][0].data
        b = test_net.params[layer][1].data
        hi = np.max(np.abs(W.flatten()))
        hi = np.sort(-np.abs(W.flatten()))[int((len(W.flatten())-1)* threshold[i])]

        # abs(val)  = 0         ==> 0
        # abs(val) >= threshold ==> 1
        interpolated = np.interp(np.abs(W), [0, hi * threshold[i], 999999999.0], [0.0, 1.0, 1.0])

        # 小于阈值的权重被随机修剪
        random_samps = np.random.rand(len(W.flatten()))
        random_samps.shape = W.shape

        # 修剪阈值
        # mask = (random_samps < interpolated)
        mask = (np.abs(W) > (np.abs(hi)))
        mask = np.bool_(mask)
        W = W * mask

        print 'non-zero W percentage = %0.5f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        # 保存修剪后的阈值
        test_net.params[layer][0].data[...] = W
        # net.params[layer][0].mask[...] = mask
        csc_W, csc_W_indx = dense_to_sparse_csc(W.flatten(), 8)
        dense_W = sparse_to_dense_csc(csc_W, csc_W_indx)
        sqarse_net[layer + '_W'] = csc_W
        sqarse_net[layer + '_W_indx'] = csc_W_indx

    # 计算修剪后的权重稀疏度
    # np.savez(model_dir + model_name +"_crc.npz",sqarse_net) # 保存存储成CRC格式的稀疏网络
    (total_percentage, percentage_list) = analyze_param(test_net, layers)
    test_loss, accuracy = test_net_accuracy(test_net)
    return (threshold, total_percentage, percentage_list, test_loss, accuracy)

def test_net_accuracy(test_net):
    test_iter = 100
    test_loss = 0
    accuracy = 0
    for test_it in range(test_iter):
        # 进行一次测试
        test_net.forward()
        # 计算test loss
        test_loss += test_net.blobs['loss'].data
        # 计算test accuracy
        accuracy += test_net.blobs['accuracy'].data

    return (test_loss / test_iter), (accuracy / test_iter)


def eval_prune_threshold(threshold_list, test_prototxt, caffemodel, prune_layers):
    def net_prune(threshold, test_prototx, caffemodel, prune_layers):
        test_net = caffe.Net(test_prototx, caffemodel, caffe.TEST)
        return prune(threshold, test_net, prune_layers)

    accuracy = []
    for threshold in threshold_list:
        results = net_prune(threshold, test_prototxt, caffemodel, prune_layers)
        print 'threshold: ', results[0]
        print '\ntotal_percentage: ', results[1]
        print '\npercentage_list: ', results[2]
        print '\ntest_loss: ', results[3]
        print '\naccuracy: ', results[4]
        accuracy.append(results[4])
    plt.plot(accuracy,'r.')
    plt.show()

# 迭代训练修剪后网络
def retrain_pruned(solver, pruned_caffemodel, threshold, prune_layers):
    #solver = caffe.SGDSolver(solver_proto)
    retrain_iter = 20

    accuracys = []
    for i in range(retrain_iter):
        solver.net.copy_from(pruned_caffemodel)
        # solver.solve()
        solver.step(500)
        _,_,_,_,accuracy=prune(threshold, solver.test_nets[0], prune_layers)
        solver.test_nets[0].save(pruned_caffemodel)
        accuracys.append(accuracy)

    plt.plot(accuracys, 'r.-')
    plt.show()


#CPU或GPU模型转换
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '../../'
#model_dir = caffe_root + 'models/SSD_300x300/'
#deploy = model_dir + 'deploy.prototxt'
#model_name = 'VGG_VOC0712_SSD_300x300_iter_60000'
#caffemodel = model_dir + model_name + '.caffemodel'

model_dir = caffe_root + 'models/mnist/'
deploy = model_dir + 'deploy.prototxt'
model_name = 'LeNet5_Mnist_shapshot_iter_10000'
caffemodel = model_dir + model_name + '.caffemodel'
test_prototxt = model_dir + 'test.prototxt'
solver_proto = model_dir + 'solver.prototxt'

solver = caffe.SGDSolver(solver_proto)

# 要修剪的层
prune_layers = ['conv1','conv2','ip1','ip2']
# 测试修剪率
test_threshold_list = [[0.3, 1 ,1 ,1], [0.4, 1 ,1 ,1], [0.5, 1 ,1 ,1], [0.6, 1 ,1 ,1], [0.7, 1 ,1 ,1],
                  [1, 0.05, 1, 1], [1, 0.1, 1, 1], [1, 0.15, 1, 1], [1, 0.2, 1, 1], [1, 0.3, 1, 1],
                  [1, 1, 0.05, 1], [1, 1, 0.1, 1], [1, 1, 0.15, 1], [1, 1, 0.2, 1], [1, 1, 0.3, 1],
                  [1, 1, 1, 0.05], [1, 1, 1, 0.1], [1, 1, 1, 0.15], [1, 1, 1, 0.2], [1, 1, 1, 0.3]]

# 验证修剪率
#eval_prune_threshold(test_threshold_list, test_prototxt, caffemodel, prune_layers)

threshold = [0.3, 0.1, 0.01, 0.2]
prune(threshold, solver.test_nets[0], prune_layers)
pruned_model = model_dir + model_name +'_pruned' + '.caffemodel'
solver.test_nets[0].save(pruned_model)

retrain_pruned(solver, pruned_model, threshold, prune_layers)



"""
# 各层对应的修剪率
threshold = [0.3, 0.1, 0.01, 0.2]
net = caffe.Net(deploy, caffemodel, caffe.TEST)
# 修剪
prune(threshold, net, prune_layers, test_prototxt)
# 保存修剪后的稀疏网络模型
output_model = model_name +'_pruned' + '.caffemodel'
net.save(output_model)
"""

```
## 量化 等
quantize.py
```py
# -*- coding:utf-8 -*-
"""
聚类量化仅仅减少内存消耗，并不能减少计算量
在实际运行中，也必须通过聚类中心表将量化后权重值转换为32位的浮点数，
因此并不能在减少网络的实际运行内存，只是减少网络的内存消耗。
要真正减少网络内存消耗，从而达到网络实际运行速度的提高，目前有两类主流方法：
    1、网络剪裁
    2、量化
网络权重共享量化也是一类重要的网络压缩方法，
其本质在于先通过聚类方法得到该层权重的聚类中心，
然后通过聚类中心值来表示原权重值。
因此权重值并不是由32位的浮点数来表示，而是由其对应的聚类中心的序号表示，
如果聚类级别为8位，此时权重值只需要用8位就能表示。
对于网络权重量化也有三个问题：
量化级别的确定，同修剪率一样，可以通过试错的试验的方法来确定
量化后网络重新训练问题
量化中心的初始选择问题：聚类中心采用线性方法初始化，将初始点均匀分散，
这种初始化方法不仅操作简单，
而且能够将对网络影响较大但实际分布较少的较大权重值也包含到初始中心点中，
因此不容易造成较大权重的丢失。
"""

# 通过Kmeans聚类的方法来量化权重
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import time

# 获得各层的量化码表
# Kmean聚类得到每层的聚类中心
# 对于Kmean聚类方法，这里调用的是scipy库的聚类函数
def kmeans_net(net, layers, num_c=16, initials=None):
    # net: 网络
    # layers: 需要量化的层
    # num_c: 各层的量化级别
    # initials: 初始聚类中心
    codebook = {} # 量化码表
    if type(num_c) == type(1):
        num_c = [num_c] * len(layers)
    else:
        assert len(num_c) == len(layers)

    # 对各层进行聚类分析
    print "==============Perform K-means============="
    for idx, layer in enumerate(layers):
        print "Eval layer:", layer
        W = net.params[layer][0].data.flatten()
        W = W[np.where(W != 0)] # 筛选不为0的权重
        # 默认情况下，聚类中心为线性分布中心
        if initials is None:  # Default: uniform sample
            min_W = np.min(W)
            max_W = np.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[idx] - 1)
            codebook[layer], _ = scv.kmeans(W, initial_uni)
        elif type(initials) == type(np.array([])):
            codebook[layer], _ = scv.kmeans(W, initials)
        elif initials == 'random':
            codebook[layer], _ = scv.kmeans(W, num_c[idx] - 1)
        else:
            raise Exception

        # 将0权重值附上
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])

    return codebook

# 随机量化权重值
def stochasitc_quantize2(W, codebook):
    # mask插入新维度：(W.shape,1)
    mask = W[:, np.newaxis] - codebook

    mask_neg = mask
    mask_neg[mask_neg > 0.0] -= 99999.0
    max_neg = np.max(mask_neg, axis=1)
    max_code = np.argmax(mask_neg, axis=1)

    mask_pos = mask
    mask_pos += 99999.0
    min_code = np.argmin(mask_pos, axis=1)
    min_pos = np.min(mask_pos, axis=1)

    rd = np.random.uniform(low=0.0, high=1.0, size=(len(W)))
    thresh = min_pos.astype(np.float32) / (min_pos - max_neg)

    max_idx = thresh < rd
    min_idx = thresh >= rd

    codes = np.zeros(W.shape)
    codes[max_idx] += min_code[max_idx]
    codes[min_idx] += max_code[min_idx]

    return codes.astype(np.int)

# 得到网络的量化权重值
def quantize_net(net, codebook):
    layers = codebook.keys()
    codes_W = {}
    print "================Perform quantization=============="
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        codes, _ = scv.vq(W.flatten(), codebook[layer]) # 根据码表得到量化权重值
        # codes = stochasitc_quantize2(W.flatten(), codebook[layer]) # 采用随机量化的方式
        codes = np.reshape(codes, W.shape)
        codes_W[layer] = np.array(codes, dtype=np.uint32)
        # 将量化后的权重保存到网络中
        W_q = np.reshape(codebook[layer][codes], W.shape)
        np.copyto(net.params[layer][0].data, W_q)

    return codes_W

# 使用聚类得到的字典进行量化各层
# 通过各层聚类来进行各层权重的量化
def quantize_net_with_dict(net, layers, codebook, use_stochastic=False, timing=False):
    start_time = time.time()
    codeDict = {} # 记录各个量化中心所处的位置
    maskCode = {} # 各层量化结果
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        if use_stochastic:
            codes = stochasitc_quantize2(W.flatten(), codebook[layer])
        else:
            codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        net.params[layer][0].data[...] = W_q

        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer] = {}
        for i in xrange(len(a)):
            codeDict[layer].setdefault(a[i], []).append(b[i])

    if timing:
        print "Update codebook time:%f" % (time.time() - start_time)

    return codeDict, maskCode

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# 重新训练及聚类中心的更新
# 重新训练时，其精度的变化图，可以看到随着迭代次数增加，其精度也逐渐提升
@static_vars(step_cache={}, step_cache2={}, count=0)
def update_codebook_net(net, codebook, codeDict, maskCode, args, update_layers=None, snapshot=None):

    start_time = time.time()
    extra_lr = args['lr'] # 基础学习速率
    decay_rate = args['decay_rate'] # 衰减速率
    momentum = args['momentum'] # 遗忘因子
    update_method = args['update'] # 更新方法
    smooth_eps = 0

    normalize_flag = args['normalize_flag'] # 是否进行归一化


    if update_method == 'rmsprop':
        extra_lr /= 100

    # 对码表与量化结果的初始化
    if update_codebook_net.count == 0:
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache
        if update_method == 'adadelta':
            for layer in update_layers:
                step_cache2[layer] = {}
                for code in xrange(1, len(codebook[layer])):
                    step_cache2[layer][code] = 0.0
            smooth_eps = 1e-8

        for layer in update_layers:
            step_cache[layer] = {}
            for code in xrange(1, len(codebook[layer])):
                step_cache[layer][code] = 0.0

        update_codebook_net.count = 1

    else:
        # 读入上次运算的结果
        step_cache2 = update_codebook_net.step_cache2
        step_cache = update_codebook_net.step_cache
        update_codebook_net.count += 1

    # 所有层名
    total_layers = net.params.keys()
    if update_layers is None: # 所有层都需要进行更新
        update_layers = total_layers

    # 权重码表的更新
    for layer in total_layers:
        if layer in update_layers:
            diff = net.params[layer][0].diff.flatten() # 误差梯度
            codeBookSize = len(codebook[layer])
            dx = np.zeros((codeBookSize)) # 编码表的误差更新
            for code in xrange(1, codeBookSize):
                indexes = codeDict[layer][code] # codeDict保存属于某编码的权重的序号
                #diff_ave = np.sum(diff[indexes]) / len(indexes)
                diff_ave = np.sum(diff[indexes]) # 统计该编码所有的误差更新和

                # 针对于不同方法进行更新
                if update_method == 'sgd':
                    dx[code] = -extra_lr * diff_ave
                elif update_method == 'momentum':
                    if code in step_cache[layer]:
                        dx[code] = momentum * step_cache[layer][code] - (1 - momentum) * extra_lr * diff_ave
                        step_cache[layer][code] = dx
                elif update_method == 'rmsprop':
                    if code in step_cache[layer]:
                        step_cache[layer][code] = decay_rate * step_cache[layer][code] + (1.0 - decay_rate) * diff_ave ** 2
                        dx[code] = -(extra_lr * diff_ave) / np.sqrt(step_cache[layer][code] + 1e-6)
                elif update_method == 'adadelta':
                    if code in step_cache[layer]:
                        step_cache[layer][code] = step_cache[layer][code] * decay_rate + (1.0 - decay_rate) * diff_ave ** 2
                        dx[code] = -np.sqrt((step_cache2[layer][code] + smooth_eps) / (step_cache[layer][code] + smooth_eps)) * diff_ave
                        step_cache2[layer][code] = step_cache2[layer][code] * decay_rate + (1.0 - decay_rate) * (dx[code] ** 2)

            # 是否需要进行归一化更新参数
            if normalize_flag:
                codebook[layer] += extra_lr * np.sqrt(np.mean(codebook[layer] ** 2)) / np.sqrt(np.mean(dx ** 2)) * dx
            else:
                codebook[layer] += dx
        else:
            pass

        # maskCode保存编码结果
        W2 = codebook[layer][maskCode[layer]]
        net.params[layer][0].data[...] = W2 # 量化后权重值

    print "Update codebook time:%f" % (time.time() - start_time)

# 保存量化结果
def store_all(net, codebook, dir_t, idx=0):
    net.save(dir_t + 'caffemodel%d' % idx)
    # 量化网络及码表
    pickle.dump(codebook, open(dir_t + 'codebook%d' % idx, 'w'))

# 恢复权重值
def recover_all(net, dir_t, idx=0):
    layers = net.params.keys()
    net.copy_from(dir_t + 'caffemodel%d' % idx)
    codebook = pickle.load(open(dir_t + 'codebook%d' % idx))
    maskCode = {}
    codeDict = {}
    for layer in layers:
        W = net.params[layer][0].data
        # 码表结果
        codes, _ = scv.vq(W.flatten(), codebook[layer])
        # 编码结果重新排列
        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer] = {}
        for i in xrange(len(a)):
            # codeDict保存每个码有哪些位置，而maskCode保存每个位置属于哪个码
            codeDict[layer].setdefault(a[i], []).append(b[i])

    return codebook, maskCode, codeDict


def analyze_log(fileName):
    data = open(fileName, "r")
    y = []
    for line in data:
        y.append(float(line.split()[0]))
    return y

# 读入测试数据
def parse_caffe_log(log):
    lines = open(log).readlines()
    try:
        res = map(lambda x: float(x.split()[-1]), lines[-3:-1])
    except Exception as e:
        print e
        res = [0.0, 0.0]
    return res

# 检测量化后网络的精度
def test_quantize_accu(test_net):
    test_iter = 100
    test_loss = 0
    accuracy = 0
    for test_it in range(test_iter):
        # 进行一次测试
        test_net.forward()
        # 计算test loss
        test_loss += test_net.blobs['loss'].data
        # 计算test accuracy
        accuracy += test_net.blobs['accuracy'].data

    return (test_loss / test_iter), (accuracy / test_iter)


def save_quantize_net(codebook, maskcode, net_filename, total_layers):
    # 编码
    quantizeNet = {}
    for layer in total_layers:
        quantizeNet[layer+'_codebook'] = np.float32(codebook[layer])
        quantizeNet[layer + '_maskcode'] = np.int8(maskcode[layer])

    np.savez(net_filename,quantizeNet)

# 保存修剪量化的网络参数
def save_pruned_quantize_net(codebook, maskcode, net_filename, total_layers):
    # W_flatten: 扁平化的权重矩阵
    # num_level: 量化级别
    quantizeNet = {}
    for layer in total_layers:
        W_flatten = maskCode[layer].flatten()
        indx = 0
        num_level = 8
        csc_W = []
        csc_indx = []
        for n in range(len(W_flatten)):
            if W_flatten[n]!=0 or indx == 2**num_level:
                csc_W.append(W_flatten[n])
                csc_indx.append(indx)
                indx = 0
            else:
                indx += 1
        if indx!=0:
            csc_W.append(0)
            csc_indx.append(indx-1)
        print max(csc_indx)
        quantizeNet[layer + '_codebook'] = np.float32(codebook[layer])
        quantizeNet[layer + '_maskcode_W'] = np.array(csc_W, dtype=np.int8)
        print max(csc_indx)
        quantizeNet[layer + '_maskcode_indx'] = np.array(csc_indx, dtype=np.int8)

    np.savez(net_filename, quantizeNet)

# caffe接口

caffe.set_mode_gpu()
caffe.set_device(0)

caffe_root = '../../'
model_dir = caffe_root + 'models/mnist/'
deploy = model_dir + 'deploy.prototxt'
solver_file = model_dir + 'solver.prototxt'
# model_name = 'LeNet5_Mnist_shapshot_iter_10000'
model_name = 'LeNet5_Mnist_shapshot_iter_10000_pruned'
caffemodel = model_dir + model_name + '.caffemodel'

dir_t = '/weight_quantize/'

# 运行测试命令
args = dict(lr=0.01, decay_rate = 0.0009, momentum = 0.9, update = 'adadelta', normalize_flag = False)

start_time = time.time()

solver = caffe.SGDSolver(solver_file)
solver.net.copy_from(caffemodel)
# 需要量化的权重
total_layers = ['conv1','conv2','ip1','ip2']

num_c = 2 ** 8 # 量化级别，由8位整数表示
codebook = kmeans_net(solver.test_nets[0], total_layers, num_c)

codeDict, maskCode = quantize_net_with_dict(solver.test_nets[0], total_layers, codebook)
quantize_net_caffemodel = model_dir + model_name + '_quantize.caffemodel'
solver.test_nets[0].save(quantize_net_caffemodel)

quantize_net_npz = model_dir + model_name + '_quantize_net'
save_pruned_quantize_net(codebook, maskCode, quantize_net_npz , total_layers)

# 迭代训练编码表
accuracys = []
co_iters = 40
ac_iters = 10
for i in xrange(2500):
    if (i % (co_iters + ac_iters) == 0 and i > 0):
        # 重新量化
        # 导入训练后的
        codebook = kmeans_net(solver.net, total_layers, num_c)
        codeDict, maskCode = quantize_net_with_dict(solver.net, total_layers, codebook)
        solver.net.save(quantize_net_caffemodel)
        solver.test_nets[0].copy_from(quantize_net_caffemodel)
        _, accu = test_quantize_accu(solver.test_nets[0])
        accuracys.append(accu)

    solver.step(1)
    if (i % (co_iters + ac_iters) < co_iters):
        # 码表更新
        update_codebook_net(solver.net, codebook, codeDict, maskCode, args=args, update_layers=total_layers)

    print "Iter:%d, Time cost:%f" % (i, time.time() - start_time)

plt.plot(accuracys, 'r.-')
plt.show()

```
