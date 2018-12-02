# opencv4.0
## C++ 11
    OpenCV 1.x 的许多C API 接口被移除，对objdetect, photo, video, videoio, imgcodecs, calib3d模块会有影响。
    OpenCV现在使用的是C++ 11，在3.x时需要将-DENABLE_CXX11=ON传递给CMake，但在4.0时默认使用C++11。
    在C++11中，标准的std :: string和std :: shared_ptr取代了手工制作的cv :: String和cv :: Ptr。现在cv::String == std::string，它cv::Ptr是轻微封装的std::shared_ptr。
    在Linux / BSD上，cv::parallel_for_现在使用std::threads而不是pthreads。

## DNN改进
    添加了基本的FP16支持（添加了新的CV_16F类型）。
    添加了对Mask-RCNN模型的支持。
    ONNX解析器已添加到OpenCV DNN模块中。它支持各种分类网络，如AlexNet，Inception v2，Resnet，VGG等，部分支持YOLO对象检测网络（YOLO的ONNX版本缺少一些提供矩形列表的最终图层）。
    API更改：默认情况下，blobFromImage方法系列不会交换红色和蓝色通道，也不会裁剪输入图像。注意：此API更改也已传播到OpenCV 3.4分支。
    修复了AMD和NVIDIA GPU上的OpenCL加速。现在可以为模型使能DNN_TARGET_OPENCL，而无需额外的环境变量。请注意，DNN_TARGET_OPENCL_FP16它仅在英特尔GPU上进行测试，因此仍需要额外的标志。
## 其它改进
    快速QR码检测器(detector)。官方计划在OpenCV 4.0正式版中添加QR码解码器(decoder)，以便有一个完整的解决方案。
    流行的Kinect Fusion算法已经实现，针对CPU和GPU（OpenCL）进行了优化，并集成到opencv_contrib / rgbd模块中。为了使实时样本有效，我们在opencv / videoio模块中更新了Kinect 2支持。
    通过所谓的“wide universal intrinsics”不断扩展SSE2，SSE4，AVX2，NEON或VSX优化内核集。
    非常高效且高质量的DIS密集光流算法已经从opencv_contrib转移到opencv的视频模块中。
    CPU和GPU加速的KinFu实时三维密集重建算法已包含在opencv_contrib中。

