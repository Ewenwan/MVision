# 工程结构 
      1. /bin             存放可执行的二进制文件
      2. /include/myslam  存放slam工程模块的头文件，只要是.h 引用头文件时时
                          需写 include "myslam/xxx.h"不容易和其他库混淆
      3. /src             存放源代码文件   主要是.cpp文件
      4. /test            存放测试用的文件 也是  .cpp文件
      5. /lib             存放编译好的库文件
      6. /config          存放配置文件
      7. /cmake_modules   存放第三方库的cmake文件 例如使用g2o eigen库时


# 数据对象结构
      0.1版本 类
      Frame     帧                Frame::Ptr  frame 
      Camera    相机模型          Camera::Ptr camera_
      MapPoint  特征点/路标点     MapPoint::Ptr map_point 
      Map       管理特征点   保存所有的特征点/路标 和关键帧
      Config    提供配置参数



#  CMakeLists.txt 编译文件
      CMAKE_MINIMUM_REQUIRED( VERSION 2.8 ) # 设定版本
      PROJECT( slam ) # 设定工程名
      SET( CMAKE_CXX_COMPILER "g++") # 设定编译器

      # 设定 可执行 二进制文件 的目录=========
      # 二进制就是可以直接运行的程序
      SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

      # 设定存放 编译出来 的库文件的目录=====
      # 库文件呢，就是为这些二进制提供函数的啦
      SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
      # 并且把该目录设为 连接目录
      LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

      # 设定头文件目录
      INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

      # 增加子文件夹，也就是进入 源代码 文件夹继续构建
      ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)
      
      # 增加一个可执行的二进制
      ADD_EXECUTABLE( main main.cpp )
      
      
# 2d 点转 3d点
```c

// 定义点云类型
typedef pcl::PointXYZRGBA PointT; # 点类型
typedef pcl::PointCloud<PointT> PointCloud;  # 点云类型

// 相机内参
const double camera_factor = 1000;  // 深度值放大倍数
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

    // 点云变量
    // 使用智能指针，创建一个空点云。这种指针用完会自动释放。
    PointCloud::Ptr cloud ( new PointCloud );
    // 遍历深度图
    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
        
        


```
