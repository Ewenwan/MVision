# 模型压缩
# 参考
[SqueezeNet-Deep-Compression](https://github.com/songhan/SqueezeNet-Deep-Compression)

# 步骤
    caffe路径
    export CAFFE_ROOT=$your_caffe_root
    解压缩模型
    python decode.py /ABSOLUTE_PATH_TO/SqueezeNet_deploy.prototxt /ABSOLUTE_PATH_TO/compressed_SqueezeNet.net /ABSOLUTE_PATH_TO/decompressed_SqueezeNet.caffemodel

    note: decompressed_SqueezeNet.caffemodel is the output, can be any name.
    测试运行
    $CAFFE_ROOT/build/tools/caffe test --model=SqueezeNet_trainval.prototxt --weights=decompressed_SqueezeNet.caffemodel --iterations=1000 --gpu 0
