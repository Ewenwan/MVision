# coco_vgg16-ssd-300-300
  coco数据集 vgg16提取特征 ssd检测框架 输入图像大小 300*300
 
## coco 数据集处理 

    1. 下载原始图片格式数据集 
        http://cocodataset.org/

    2. 下载数据库api
        git clone https://github.com/weiliu89/coco.git cocoapi
        cd cocoapi
        git checkout dev
        
    3. 编译数据库api ----可能不需要
        cd cocoapi/PythonAPI
        python setup.py build_ext --inplace
        
    4. 标注文件处理 ------可能不需要
        # Check scripts/batch_split_annotation.py and change settings accordingly.
        python scripts/batch_split_annotation.py
        # Create the minival2014_name_size.txt and test-dev2015_name_size.txt in $CAFFE_ROOT/data/coco
        python scripts/batch_get_image_size.py
        
        图片文件列表文件生成 训练和测试txt 可参考darknet
        
    5. 生成数据库文件
        生成图像列表文件：
            python data/coco/create_list.py------可能也用不到------ 
            生成子集  minival.txt, testdev.txt, test.txt, train.txt
            
        生成对应的数据库：
            ./data/coco/create_data.sh
            得到：
                #   - $HOME/data/coco/lmdb/coco_minival_lmdb
                #   - $HOME/data/coco/lmdb/coco_testdev_lmdb
                #   - $HOME/data/coco/lmdb/coco_test_lmdb
                #   - $HOME/data/coco/lmdb/coco_train_lmdb
