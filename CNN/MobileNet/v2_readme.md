# MobileNet V2 with Tensorflow
论文链接 https://arxiv.org/abs/1801.04381
[来源](https://github.com/neuleaf/MobileNetV2)

## Task1 classification
1. 制作tfrecord文件
   
   训练图片及类别标签组织为`train_data.txt`的形式，每一行表示一个样本及其标签，用空格或Tab隔开.
   
   run `python convert2tf.py [--options]` 生成tfrecord文件。
   （同时会打印数据集样本的格式'num_samples', 训练时需要输入该参数）

2. 训练
   
   run `python main.py --epoch 10 --dataset_dir ./tfrecords --n_classes 10 --batch_size 4 --num_samples 20000 [--options]`启动训练.
   注意更改参数
3. test/infer

    测试时需要指定`--no_train`，如：
   `python main.py --n_classe 10 --no_train`

## Task2 detection

TODO
## Task3 segmentation

TODO
