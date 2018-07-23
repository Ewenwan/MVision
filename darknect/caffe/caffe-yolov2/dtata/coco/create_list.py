# -*- coding:utf-8 -*-
import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys

#HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

# If true, re-create all list files.
redo = True
# The root directory which holds all information of the dataset.
# coco数据集
# data_dir = "{}/data/coco".format(HOMEDIR)
data_dir = "./coco2017/coco"
# The directory name which holds the image sets.
# 图片id 列表
imgset_dir = "ImageSets"
# The direcotry which contains the images.
# 图片文件夹
img_dir = "Images"
# 图片格式
img_ext = "jpg"
# The directory which contains the annotations.
# json标注文件地址
# anno_dir = "Annotations"
# xml标注文件地址
anno_dir = "annotations"
# ison格式 
#anno_ext = "json"
# xml 格式标注文件
anno_ext = "xml"

# 生成的列表文件
train_list_file = "{}/train2017_img_xml_label.txt".format(CURDIR)
#minival_list_file = "{}/minival.txt".format(CURDIR)
#testdev_list_file = "{}/testdev.txt".format(CURDIR)
val_list_file   = "{}/val2017_img_xml_label.txt".format(CURDIR)

# Create training set.
# We follow Ross Girschick's split.

# 训练集
if redo or not os.path.exists(train_list_file):
    datasets = ["train2017"]
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                #subset = name.split("_")[1]
                subset = "train2017"
                img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
                #assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                #        "{}/{} does not exist".format(data_dir, img_file)
                #if (~os.path.exists("{}/{}".format(data_dir, img_file))):
                #    print("{}/{} does not exist".format(data_dir, img_file))
                #    continue;
                
                
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                #assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                #        "{}/{} does not exist".format(data_dir, anno_file)
                #if (~os.path.exists("{}/{}".format(data_dir, anno_file))):
                #    print("{}/{} does not exist".format(data_dir, anno_file))
                #    continue;
                
                img_files.append(img_file)
                anno_files.append(anno_file)
    # Shuffle the images.
    idx = [i for i in xrange(len(img_files))]
    shuffle(idx)
    with open(train_list_file, "w") as f:
        for i in idx:
            f.write("{} {}\n".format(img_files[i], anno_files[i]))
# 验证
if redo or not os.path.exists(val_list_file):
    datasets = ["val2017"]
    subset = "val2017"
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
                #assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                #        "{}/{} does not exist".format(data_dir, img_file)
                #if (~os.path.exists("{}/{}".format(data_dir, img_file))):
                #    print("{}/{} does not exist".format(data_dir, img_file))
                #    continue;
                
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                #assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                #        "{}/{} does not exist".format(data_dir, anno_file)
                #if (~os.path.exists("{}/{}".format(data_dir, anno_file))):
                #    print("{}/{} does not exist".format(data_dir, anno_file))
                #    continue;
                
                img_files.append(img_file)
                anno_files.append(anno_file)
    with open(val_list_file, "w") as f:
        for i in xrange(len(img_files)):
            f.write("{} {}\n".format(img_files[i], anno_files[i]))

