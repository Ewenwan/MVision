#-*- coding: utf-8 -*-
# 配置文件 先验框 和 类别名称
"""
Yolov2 anchors and coco classes
"""

# 聚类出来的 5中先验框长宽
"""
anchors = [[0.738768, 0.874946],
           [2.42204, 2.65704],
           [4.30971, 7.04493],
           [10.246, 4.59428],
           [12.6868, 11.8741]]
"""
anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]
# coco 数据集 标签类别 名称
def read_coco_labels():
    f = open("./data/coco_classes.txt")
    class_names = []
    for l in f.readlines():#每一行
        class_names.append(l[:-1])#加入到 类名 列表
    return class_names

class_names = read_coco_labels()
