# coding=utf-8

# 使用说明
# 将该文件放在ILSVRC2016/bject_detection/ILSVRC目录下，并将Data文件夹重命名为JPEGImages
# 执行该工具，Lists目录下会生成图片路径列表
# labels目录下会生成yolo需要的标注文件

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil


# 获取所有包含标注文件的的目录路径
def get_dirs():
    dirs = ['DET/train/ILSVRC2014_train_0006', 'DET/train/ILSVRC2014_train_0005', 'DET/train/ILSVRC2014_train_0004',
            'DET/train/ILSVRC2014_train_0003', 'DET/train/ILSVRC2014_train_0002', 'DET/train/ILSVRC2014_train_0001',
            'DET/train/ILSVRC2014_train_0000', 'DET/val']
    dirs_2013 = os.listdir('JPEGImages/DET/train/ILSVRC2013_train/')
    for dir_2013 in dirs_2013:
        dirs.append('DET/train/ILSVRC2013_train/' + dir_2013)
    return dirs


# 获取所需要的类名和id
# path为类名和id的对应关系列表的地址（标注文件中可能有很多类，我们只加载该path指向文件中的类）
# 返回值是一个字典，键名是类名，键值是id
def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        D[temp[1]] = temp[0]#类别名
    return D


# 将ROI的坐标转换为yolo需要的坐标
# size是图片的w和h
# box里保存的是ROI的坐标（左上角坐标值(x,y)和右下角坐标值）
# 返回值为ROI中心点相对于图片大小的比例坐标（0~1），
# 和ROI的w、h相对于图片大小的比例（0~1）
def convert(size, box):
    dw = 1. / size[0]# 图片的w 宽  的分辨率   1/418
    dh = 1. / size[1]# 图片的h 高
    x = (box[0] + box[2]) / 2.0#中心点坐标
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)# 比例(0~1)


# 将labelImg 生成的xml文件转换为yolo需要的txt文件
# image_dir 图片所在的目录的路径
# image_id图片名
def convert_annotation(image_dir, image_id):
    in_file = open('Annotations/%s/%s.xml' % (image_dir, image_id))#每一个图像的 xml文件
    obj_num = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)# 宽度
    h = int(size.find('height').text)# 高度

    for obj in root.iter('object'):# 多个物体的框
        cls = obj.find('name').text# 物体类别
        if cls not in classes:# 跳过没有的类别
            continue
        obj_num = obj_num + 1# 目标数量++
        if obj_num == 1:#该 图片中的 目标数量
            out_file = open('labels/%s/%s.txt' % (image_dir, image_id), 'w')
        cls_id = classes[cls]  # 获取该类物体在yolo训练中的id
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))#这里对应的是  左上角坐标值(x,y)和右下角坐标值
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    if obj_num > 0:#包含多个物体
        list_file = open('Lists/%s.txt' % image_dir.split('/')[-1], 'a')  # 数据集的图片list保存路径
        list_file.write('%s/JPEGImages/%s/%s.JPEG\n' % (wd, image_dir, image_id))
        list_file.close()

# 子字符串
def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


# 获取FindPath路径下指定格式（FlagStr）的文件名（不包含后缀名）列表
def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if (len(FileNames) > 0):
        for fn in FileNames:
            if (len(FlagStr) > 0):
                if (IsSubString(FlagStr, fn)):
                    FileList.append(fn[:-4])
            else:
                FileList.append(fn)

    if (len(FileList) > 0):
        FileList.sort()

    return FileList

# image_net 数据集 类别名
image_net_class_file = '/darknet/data/imagenet_list.txt'
classes = get_classes_and_index(image_net_class_file )
dirs = get_dirs()

wd = getcwd()# 当前路径

# Lists 目录若不存在，创建Lists目录。若存在，则清空目录
if not os.path.exists('Lists/'):
    os.makedirs('Lists/')
else:
    shutil.rmtree('Lists/')
    os.makedirs('Lists/')

for image_dir in dirs:
    if not os.path.exists('JPEGImages/' + image_dir):
        print("JPEGImages/%s dir not exist" % image_dir)
        continue
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists('labels/%s' % (image_dir)):
        os.makedirs('labels/%s' % (image_dir))
    else:
        shutil.rmtree('labels/%s' % (image_dir))
        os.makedirs('labels/%s' % (image_dir))
    image_ids = GetFileList('Annotations/' + image_dir, ['xml'])
    for image_id in image_ids:
        print(image_id)
        convert_annotation(image_dir, image_id)
