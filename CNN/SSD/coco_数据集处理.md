# coco数据集 20G

    超过 20万 张图片，80种物体类别，图片大小不一，3通道彩色图像。
    所有的物体实例都用详细的分割mask进行了标注，
    共标注了超过 50万 个目标实体.
    
下载数据集   

    1. 下载 数据库API
     git clone https://github.com/pdollar/coco
     cd coco
    2. 创建 images文件夹 并下载 图像数据 解压
     在images文件夹下下载  点击链接可直接下载
     wget -c https://pjreddie.com/media/files/train2014.zip    12.5G
     wget -c https://pjreddie.com/media/files/val2014.zip      6.18G

     解压
     unzip -q train2014.zip
     unzip -q val2014.zip
    3. 下载标注文件等
      cd ..
      wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
      wget -c https://pjreddie.com/media/files/coco/5k.part
      wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
      wget -c https://pjreddie.com/media/files/coco/labels.tgz
      sudo tar xzf labels.tgz                        标签
      sudo unzip -q instances_train-val2014.zip     分割  得到 annotations  实例分割

      生成训练/测试图像列表文件  darknet框架下格式
      paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt   测试验证数据
      paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt  训练数据
    4.  caffe lmdb格式转换
      2014年
      train2014.zip 训练集图片
      val2014.zip   验证集图片
      instances_train-val2014.zip  总的json标签
      labels.tgz    darknet下 的 txt标签
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
            
        创建图片地址+标签地址的 列表文件    
            python data/coco/create_list.py
        生成lmdb 文件 and make soft links at examples/coco/
           ./create_data.sh
            
            
