# caffe yolov2 实现

[代码](https://github.com/Ewenwan/caffe-yolov2)


## 完全从0开始训练，效果不好

## 可以使用 yolo_v2.weights 转到caffemodel下，基于这个参数进行训练，会好很多 在good目录下


# BOX data   coco数据集生成

  2017年 
    http://images.cocodataset.org/zips/train2017.zip 训练集图片
    http://images.cocodataset.org/zips/val2017.zip   验证集图片
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip 总的json标签

    处理 ：
    下载 coco数据库处理脚本 
        git clone https://github.com/weiliu89/coco.git
        cd coco
        git checkout dev  # 必要 在 PythonAPI 或出现 scripts/ 文件夹 一些处理脚本
    安装：
        cd coco/PythonAPI
        python setup.py build_ext --inplace
        
    将总的json文件拆分成 各个图像的json
        python scripts/batch_split_annotation.py 
        
    获取 图片id 对应的图片尺寸大小 长宽
        python scripts/batch_get_image_size.py
        
    创建图片地址+ json标签地址的 列表文件    
        python data/coco/create_list.py
    生成lmdb 文件 and make soft links at examples/coco/
       ./create_data.sh
    
    
    2：
    将各个图像的json 转成 voc格式的 xml文件
    python Construct_XML_Annotations_from_COCO_JSON.py
    
    创建图片地址+ xml标签地址的 列表文件    
        python data/coco/create_list_xml.py
    生成lmdb 文件 and make soft links at examples/coco/
       ./convert.sh 
       
       
