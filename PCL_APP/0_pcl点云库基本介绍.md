# pcl 简介 
      PCL（Point Cloud Library）是在吸收了前人点云相关研究基础上
      建立起来的大型跨平台开源C++编程库，
      它实现了大量点云相关的通用算法和高效数据结构，
      
      涉及到
            点云获取、
            滤波、
            分割、
            配准、
            检索、
            特征提取、
            识别、
            追踪、
            曲面重建、
            可视化等。
            
      支持多种操作系统平台，可在Windows、Linux、Android、Mac OS X、部分嵌入式实时系统上运行。
      
      如果说OpenCV是2D信息获取与处理的结晶，
      
      那么PCL就在3D信息获取与处理上具有同等地位，PCL是BSD授权方式，
      
      可以免费进行商业和学术应用。

# PCL的发展与创景

      PCL起初是ROS（Robot Operating System）下由来自于
      慕尼黑大学（TUM - Technische Universität München）
      和斯坦福大学（Stanford University）Radu博士等人维护和开发的开源项目，
      主要应用于机器人研究应用领域，
      随着各个算法模块的积累，于2011年独立出来，
      正式与全球3D信息获取、处理的同行一起，组建了强大的开发维护团队，
      以多所知名大学、研究所和相关硬件、软件公司为主，可参考图1。


# PCL在机器人领域中的应用

      移动机器人对其工作环境的有效感知、辨识与认知，
      是其进行自主行为优化并可靠完成所承担任务的前提和基础。
      
      如何实现场景中物体的　有效分类　与　识别是移动机器人场景认知的核心问题，
      目前基于　视觉图像处理技术　来进行场景的认知是该领域的重要方法。
      
      但移动机器人在线获取的视觉图像质量受光线变化影响较大，
      特别是在光线较暗的场景更难以应用，随着RGBD获取设备的大量推广，在机器人领域势必掀起一股深度信息
      
      结合2D信息的应用研究热潮，深度信息的引入能够使机器人更好地对环境进行认知、辨识，
      
      与图像信息在机器人领域的应用一样，需要强大智能软件算法支撑，PCL就为此而生，
      
      最重要的是PCL本身就是为机器人而发起的开源项目，PCL中不仅提供了对现有的RGBD信息的获取设备的支持，
      
      还提供了高效的　分割、特征提取、识别、追踪等最新的算法，
      最重要的是它可以移植到android、ubuntu等主流Linux平台上，
      PCL无疑将会成为机器人应用领域一把瑞士军刀。

# PCL在　虚拟现实、人机交互中的应用

      虚拟现实技术（简称VR），又称灵境技术，
      是以沉浸性、交互性和构想性为基本特征的计算机高级人机界面。
      它综合利用了
            计算机图形学、
            仿真技术、
            多媒体技术、
            人工智能技术、
            计算机网络技术、
            并行处理技术和
            多传感器技术，
            
      模拟人的视觉、听觉、触觉等感觉器官功能，
      使人能够沉浸在计算机生成的虚拟境界中，并能够通过语言、
      手势等自然的方式与之进行实时交互，
      创建了一种适人化的多维信息空间，具有广阔的应用前景。
      目前各种交互式体感应用的推出，
      让虚拟现实与人机交互发展非常迅速，以微软、华硕、三星等为例，
      目前诸多公司推出的RGBD解决方案，势必会让虚拟现实走出实验室，
      因为现有的RGBD设备已经开始大量推向市场，
      只是缺少，其他应用的跟进，这正是在为虚拟现实和人机交互应用铸造生态链的底部，
      笔者认为这也正是PCL为何在此时才把自己与世人分享的重要原因所在，
      它将是基于RGBD设备的虚拟现实和人机交互应用生态链中最重要的一个环节。
      让我们抓住这一个节点，立足于交互式应用的一片小天地，但愿本书来的不是太迟。

# PCL特点
      PCL利用OpenMP、GPU、CUDA等先进高性能计算技术，通过并行化提高程序实时性。
      K近邻搜索操作的构架是基于FLANN (Fast Library for Approximate Nearest Neighbors)
      所实现的，速度也是目前技术中最快的。
      PCL中的所有模块和算法都是通过Boost共享指针来传送数据的，
      因而避免了多次复制系统中已存在的数据的需要，
      从0.6版本开始，PCL就已经被移入到Windows，MacOS和Linux系统，
      并且在Android系统也已经开始投入使用，这使得PCL的应用容易移植与多方发布。

      ========================================
# 在PCL中一个处理管道的基本接口程序是：
      创建处理对象：（例如过滤、特征估计、分割等）;
      使用setInputCloud通过输入点云数据，处理模块;
      设置算法相关参数;
      调用计算（或过滤、分割等）得到输出。

      为了进一步简化和开发，PCL被分成一系列较小的代码库，
      使其模块化，以便能够单独编译使用提高可配置性，特别适用于嵌入式处理中:

       1. libpcl filters： 
            如采样、去除离群点、下采样(特征提取)、拟合估计等数据实现过滤器；
       2. libpcl features：
            实现多种三维特征，如曲面法线、曲率、边界点估计、矩不变量、主曲率，
            PFH和FPFH特征，旋转图像、积分图像，NARF描述子，
            RIFT，相对标准偏差，数据强度的筛选等等；
       3. libpcl I/O：     
            实现数据的输入和输出操作，例如点云数据文件（PCD）的读写；
       4. libpcl segmentation：
            实现聚类提取，如通过采样一致性方法对一系列参数模型（如平面、柱面、球面、直线等）
            进行模型拟合点云分割提取，提取多边形棱镜内部点云等等；
       5. libpcl surface：     
            实现表面重建技术，如网格重建、凸包重建、移动最小二乘法平滑等；
       6. libpcl register：   
            实现点云配准方法，如ICP,NDT等；
       7. libpclkeypoints：   
            实现不同的关键点的提取方法，这可以用来作为预处理步骤，决定在哪儿提取特征描述符；
       8. libpcl range ：      
            实现支持不同点云数据集生成的范围图像。
            
# PCL命令行安装　　编译好的二进制文件     
      仓库
      sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
      sudo apt-get update
      sudo apt-get install libpcl-all
      
# 源码安装
      安装依赖:
            Boost，Eigen，FlANN，VTK，OpenNI，QHull
            sudo apt-get install　build-essential　libboost-all-dev　
            sudo apt-get install libvtk5-dev 

## Vtk，（visualization toolkit 可视化工具包）是一个开源的免费软件系统，

[Vtk教程](http://blog.csdn.net/www_doling_net/article/details/8763686)

      主要用于三维计算机图形学、图像处理和可视化。
      它在三维函数库OpenGL的基础上采用面向对象的设计方法发展而来，且具有跨平台的特性。 
      Vtk是在面向对象原理的基础上设计和实现的，它的内核是用C++构建的
      VTK面向对象，含有大量的对象模型。 
      源对象是可视化流水线的起点，

      映射器（Mapper）对象是可视化流水线的终点，是图形模型和可视化模型之间的接口. 
      回调（或用户方法）: 观察者监控一个对象所有被调用的事件，
      如果正在监控的一个事件被触发，一个与之相应的回调函数就会被调用。

      图形模型：
      Renderer 渲染器，vtkRenderWindow 渲染窗口

      可视化模型：
      vtkDataObject 可以被看作是一个二进制大块（blob）
      vtkProcessObject 过程对象一般也称为过滤器，按照某种运算法则对数据对象进行处理


## FLANN介绍
      FLANN库全称是Fast Library for Approximate Nearest Neighbors，
      它是目前最完整的（近似）最近邻开源库。
[最近邻开源库 论文](http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf)
[安装下载](http://www.cs.ubc.ca/research/flann/#download)

## Eigen安装　c++　矩阵运算库　几何学　坐标变换　其次坐标
      linux下安装
      sudo apt-get install libeigen3-dev
      定位安装位置
      locate eigen3
      sudo updatedb

## 下载pcl源码 源码安装

      git clone https://github.com/PointCloudLibrary/pcl pcl-trunk

      //注意　PCL_ROS 其实引用了PCL库，不要随意编译PCL库，可能导致PCL-ROS不能使用！
      // PCL自动安装的时候与C11不兼容，如果想使用C11,需要自己编译PCL库，
      //并在PCL编译之前的CMakelist.txt中加入C11的编译项！

      //所以　ros项目中如果使用　支持　c++11　
      // 那么使用pcl时pcl必须源码编译，并且需要修改pcl源码的　CMakelist.txt　 加入支持c++11的选项
      //# 添加c++ 11标准支持
      //set( CMAKE_CXX_FLAGS "-std=c++11" )

      cd pcl-trunk && mkdir build && cd build
      cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
      make -j2
      sudo make -j2 install

# 对象
      ====================================
      // .width 和 .height 对象 int
      cloud.width = 640; // there are 640 points per line 每一行 640个点
      cloud.height = 480; //  640*480=307200  个点
      =============================
      cloud.width = 307200;
      cloud.height = 1; // 307200 个点
      ================================
      // .points 对象为 点容器 std::vector<pcl::PointXYZ>
      pcl::PointCloud<pcl::PointXYZ> cloud;
      std::vector<pcl::PointXYZ> data = cloud.points;
      =====================
      // .is_dense 对象 bool
      查看是否包含 Inf/NaN 值 有的话为false
      ===============================
      // .sensor_origin_ (Eigen::Vector4f)  
      传感器记录的 位姿

      =======================
      struct PointXYZ
      {
        float x;
        float y;
        float z;
        float padding;
      };

      PCL中可用的PointT类型：
      PointXYZ——成员变量：float x,y,z;
           PointXYZ是使用最常见的一个点数据类型，因为他之包含三维XYZ坐标信息，
      这三个浮点数附加一个浮点数来满足存储对齐，
      可以通过points[i].data[0]或points[i].x访问点X的坐标值
      union
      {
        float data[4];
        struct
        {
          float x;
          float y;
          float z;
        };
      };

      PointNormal

      PointXYZI——成员变量：float x,y,z,intensity。
        PointXYZI是一个简单的X Y Z坐标加intensity的point类型

      PointXYZRGBA——成员变量：float x,y,z;uint32_t  rgba 
      PointXYZRGB——float x,y,z,rgb   除了RGB信息被包含在一个浮点数据变量中，其他的和 PointXYZRGBA

      PointXY——成员变量：float x,y        简单的二维x-y结构代码

      InterestPoint——成员变量：float x,y,z,strength除了strength表示关键点的强度测量值，其他的和PointXYZI

      Normal——成员变量：float normal[3],curvature;
        另一个常用的数据类型，
        Normal结构体表示给定点所在样本曲面上的法线方向，
        以及对应曲率的测量值，
        例如访问法向量的第一个坐标可以通过points[i].data_n[0]或者points[i].normal[0]或者points[i]

      PointNormal——成员变量：float x,y,z;   float normal[3] ,curvature ; 
        PointNormal是存储XYZ数据的point结构体，并且包括了采样点的法线和曲率

      [点基础类型参考](https://blog.csdn.net/u013019296/article/details/70052287)

      ===========================

      PCLPointCloud2 () : header (), height (0), width (0), fields (),
       is_bigendian (false), point_step (0), row_step (0),
        data (), is_dense (false)

      PCLPointCloud2 >>>>>> pcl::PointXYZ

        // 转换为模板点云
        pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);


      cloud_filtered_blob 声明的数据格式为
      pcl::PCLPointCloud2::Ptr  cloud_filtered_blob (new pcl::PCLPointCloud2);

      cloud_filtered 申明的数据格式  
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>)

      ====================================

      const sensor_msgs::PointCloud2ConstPtr  input
       // 创建一个输出的数据格式
        sensor_msgs::PointCloud2 output;  //ROS中点云的数据格式
        //对数据进行处理
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        output = *input;
        pcl::fromROSMsg(output,*cloud);



## PCD（点云数据）文件格式，以下几种格式
      （1）PLY是一种多边形文件格式，
      （2）STL是3D System公司创建的模型文件格式，主要应用于CAD，CAM领域
      （3）OBJ是从几何学上定义的文件格式，
      （4）X3D是符合ISO标准的基于XML的文件格式，表示3D计算机图形数据


      PCD文件头格式

      每个PCD文件包含一个文件头，确定和声明文件中存储这点云的数据的某种特性，PCD文件必须用ASCII码来编码，

         （1）VERSION---------指定PCD文件版本
         （2） FIELSS------------指定一个点恶意有的每一个维度和字段的名字例如
                 FILEDS  x y z                             #XYZ data
                 FILEDS x y z rgb                          #XYZ + color
                 FILEDS x y z normal_x normal_y normal_z   #XYZ +surface normal
                 FILEDS j1 j2 j3                           #moment invariants 距
         (3) SIZE-----------用字节数指定每一个维度的大小        例如
                         unsigned  char/char?     has  1 byte
                         unsigned  short/short?   has  2 byte
                         double  ?                has  8 byte
         (4) TYPE------------用一个字符指定每一个维度的类型  被接受类型有
                   I----------------表示有符号类型   int8(char)   int16 (short)     int32(int)
                   U----------------表示无符号类型   ------------------
                   F----------------表示浮点类型
      （5）COUNT----------指定每一维度包含的元数目（默认情况下，没有设置 的话，所有维度的数目被设置为1）
      （6）WIDTH------用点的数量表示点云数据集的宽度，根据有序点云还是无序点云，WIDTH有两层解释：
                    1，它能确定无序数据集的点云中点的个数，
                    2，它能确定有序点云数据集的宽度
            注意有序点云数据集，意味着点云是类似与图像的结构，数据分为行和列，
            这种点云的实例包括立体摄像机和时间飞行摄像机生成的数据，
            有序数据集的优势在于，预先了解相邻点（和像素点类似）的关系，邻域操作更加高效，
            这样就加速了计算并降低了PCL中某些算法的成本。
            例如：WIDTH   640    #每行有640个点

      （7）HEIGHT---------------用点的数目表示点云数据集的高度。类似于WIDTH也有两层解释，
        有序点云的例子：WIDTH    640            #像图像一样的有序结构，有640行480列，
                 HEIGHT   480            #这样该数据集中共有640*480=307200个人点
        无序点云例子：
                WIDTH  307200
                HEIGHT    1              #有307200个点的无序点云数据集
      （8）VIEWPOINT--------------------指定数据集中点云的获取视角。
          VIEWPOINT有可能在不同坐标系之间转换的时候应用，在辅助获取其他特征时，也比较有用，  
        例如曲面发现，在判断方向一致性时，需要知道视点的方位
              视点信息被指为
            平移（tx ty tz）  + 四元数（qw qx qy qz 表示旋转方向）
      （9 ) POINTS----------------指定点云中点的总数
      （10） DATA---------------指定存储点云数据的数据结构，有两种形式：ASCII和二进制
                （1）如果易ASCII形式，每一点占据一个新行，
                （2）如果以二进制的形式，这里数据是数组向量的PCL

      （注意PCD文件的文件头部分必须是以上部分顺序的精确的指定）



          # PCD  v.7  --Point Cloud Data file format  
          VERSION .7  
          FIELDS x y z rgb  
          SIZE 4 4 4 4   
          TYPE F FFF  
          COUNT 1 1 1 1  
          WIDTH 213  
          HEIGHT 1  
          VIEWPOINT 0 0 0  1  0 0 0   
          POINTS 213  
          DATA ascii  
          0.93773    0.33763 0 4.218e+06  
          0.90805   0.35641  0 4.2108e+06  


      ======================
##  模板类实现
      // foo.h
      #ifndef PCL_FOO_
      #define PCL_FOO_

      template <typename PointT>
      class Foo
      {
        public:
          void
          compute (const pcl::PointCloud<PointT> &input,
                   pcl::PointCloud<PointT> &output);
      }
      #endif // PCL_FOO_
      ===================================
      // impl/foo.hpp
      #ifndef PCL_IMPL_FOO_
      #define PCL_IMPL_FOO_
      #include "foo.h"
      template <typename PointT> void
      Foo::compute (const pcl::PointCloud<PointT> &input,
                    pcl::PointCloud<PointT> &output)
      {
        output = input;
      }
      #endif // PCL_IMPL_FOO_


      =============================
      // foo.cpp
      #include "pcl/point_types.h"
      #include "pcl/impl/instantiate.hpp"
      #include "foo.h"
      #include "impl/foo.hpp"
      // Instantiations of specific point types
      PCL_INSTANTIATE(Foo, PCL_XYZ_POINT_TYPES));


      ============================================
      // 定义自己的  点类型

      #define PCL_NO_PRECOMPILE
      #include <pcl/point_types.h>
      #include <pcl/point_cloud.h>
      #include <pcl/io/pcd_io.h>

      struct MyPointType
      {
        PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
        float test;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
      } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

      POINT_CLOUD_REGISTER_POINT_STRUCT (MyPointType,           // here we assume a XYZ + "test" (as fields)
                                         (float, x, x)
                                         (float, y, y)
                                         (float, z, z)
                                         (float, test, test)
      )

      int
      main (int argc, char** argv)
      {
        pcl::PointCloud<MyPointType> cloud;
        cloud.points.resize (2);
        cloud.width = 2;
        cloud.height = 1;

        cloud.points[0].test = 1;
        cloud.points[1].test = 2;
        cloud.points[0].x = cloud.points[0].y = cloud.points[0].z = 0;
        cloud.points[1].x = cloud.points[1].y = cloud.points[1].z = 3;

        pcl::io::savePCDFile ("test.pcd", cloud);
      }

      =================================================
##  定义自己的算法类  
       双边滤波（Bilateral filter）  示例
      http://pointclouds.org/documentation/tutorials/writing_new_classes.php#writing-new-classes

      =============================
      CMakeList.txt

      cmake_minimum_required ( VERSION 2.6 FATAL_ERROR)   #对于cmake版本的最低版本的要求
      project(ch2)                                        #建立的工程名，例如源代码目录路径的变量名为CH_DIR
                                                          #工程存储目录变量名为CH_BINARY_DIR
      #要求工程依赖的PCL最低版本为1.3，并且版本至少包含common和IO两个模块  这里的REQUIRED意味着如果对应的库找不到 则CMake配置的过程将完全失败，
      #因为PCL是模块化的，也可以如下操作：
      #           一个组件  find_package(PCL 1.6 REQUIRED COMPONENTS  io)
      #           多个组件  find_package(PCL 1.6 REQUIRED COMPONENTS commom io)
      #           所有组件  find_package(PCL 1.6 REQUIRED )                    
      find_package(PCL 1.6 REQUIRED)  


      #下面的语句是利用CMake的宏完成对PCL的头文件路径和链接路径变量的配置和添加，如果缺少下面几行，生成文件的过程中就会提示
      #找不到相关的头文件，在配置CMake时，当找到了安装的PCL，下面相关的包含的头文件，链接库，路径变量就会自动设置
      #                    PCL_FOUND:          如果找到了就会被设置为1 ，否则就不设置
      #                    PCL_INCLUDE_DIRS:   被设置为PCL安装的头文件和依赖头文件的目录
      #                    PCL_LIBRARIES:      被设置成所建立和安装的PCL库头文件
      #                    PCL_LIBRARIES_DIRS: 被设置成PCL库和第三方依赖的头文件所在的目录
      #                    PCL_VERSION:        所找到的PCL的版本
      #                    PCL_COMPONENTS:     列出所有可用的组件
      #                    PCL_DEFINITIONS:    列出所需要的预处理器定义和编译器标志
      include_directories(${PCL_INCLUDE_DIRS})
      link_directories(${PCL_LIBRARIES_DIRS})
      add_definitions(${PCL_DEFINITIONS})

      #这句话告诉CMake从单个源文件write_pcd建立一个可执行文件
      add_executable(write_pcd write_pcd.cpp)
      #虽然包含了PCL的头文件，因此编译器知道我们现在访问所用的方法，我们也需要让链接器知道所链接的库，PCL找到库文件由
      #PCL_COMMON_LIBRARIES变量指示，通过target_link_libraries这个宏来出发链接操作
      target_link_libraries(write_pcd ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

