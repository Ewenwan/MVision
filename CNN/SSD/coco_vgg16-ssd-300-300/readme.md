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
# darknet 那边的处理
    1. 下载 数据库API
     git clone https://github.com/pdollar/coco
     cd coco
    2. 创建 images文件夹 并下载 图像数据 解压
     在images文件夹下下载  点击链接可直接下载
     wget -c https://pjreddie.com/media/files/train2014.zip
     wget -c https://pjreddie.com/media/files/val2014.zip

     解压
     unzip -q train2014.zip
     unzip -q val2014.zip
    3. 下载标注文件等
      cd ..
      wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
      wget -c https://pjreddie.com/media/files/coco/5k.part
      wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
      wget -c https://pjreddie.com/media/files/coco/labels.tgz
      sudo tar xzf labels.tgz                        标签
      sudo unzip -q instances_train-val2014.zip     分割  得到 annotations  实例分割

      生成训练/测试图像列表文件
      paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt   测试验证数据
      paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt  训练数据
      
    4. caffe下的后处理
      生成 文件名
      训练集中的 val 部分
      cat trainvalno5k.txt | awk -F "/" '{for(i=1;i<=NF;i++){ if(i==11) print $i; }}' | grep val | cut -b 1-25 | tee train_val_list.txt
      训练集的 train 部分
      cat trainvalno5k.txt | awk -F "/" '{for(i=1;i<=NF;i++){ if(i==11) print $i; }}' | grep tra | cut -b 1-27 | tee train_list.txt
      
      验证集中的 验证 部分
      cat 5k.txt | awk -F "/" '{for(i=1;i<=NF;i++){ if(i==11) print $i; }}' | cut -b 1-25 | tee val_list.txt
      
      
      
