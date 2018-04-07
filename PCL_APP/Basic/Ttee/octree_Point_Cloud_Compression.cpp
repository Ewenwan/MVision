/*
八叉树空间分割进行点分布区域的压缩
通过对连续帧之间的数据相关分析，检测出重复的点云，并将其去除掉再进行传输.

点云由庞大的数据集组成，这些数据集通过距离、颜色、法线等附加信息来描述空间三维点。
此外，点云能以非常高的速率被创建出来，因此需要占用相当大的存储资源，
一旦点云需要存储或者通过速率受限制的通信信道进行传输，提供针对这种数据的压缩方法就变得十分有用。
PCL库提供了点云压缩功能，它允许编码压缩所有类型的点云，包括“无序”点云，
它具有无参考点和变化的点尺寸、分辨率、分布密度和点顺序等结构特征。
而且，底层的octree数据结构允许从几个输入源高效地合并点云数据
.

Octree八插树是一种用于描述三维空间的树状数据结构。
八叉树的每个节点表示一个正方体的体积元素，每个节点有八个子节点，
将八个子节点所表示的体积元素加在一起就等于父节点的体积。
Octree模型：又称为八叉树模型，若不为空树的话，
树中任一节点的子节点恰好只会有八个，或零个，也就是子节点不会有0与8以外的数目。

Log8(房间内的所有物品数)的时间内就可找到金币。
因此，八叉树就是用在3D空间中的场景管理，可以很快地知道物体在3D场景中的位置，
或侦测与其它物体是否有碰撞以及是否在可视范围内。

*/

#include <pcl/point_cloud.h>                         // 点云类型
#include <pcl/point_types.h>                          //点数据类型
#include <pcl/io/openni_grabber.h>                    //点云获取接口类
#include <pcl/visualization/cloud_viewer.h>            //点云可视化类

#include <pcl/compression/octree_pointcloud_compression.h>//点云压缩

#include <stdio.h>
#include <sstream>
#include <stdlib.h>

#ifdef WIN32
# define sleep(x) Sleep((x)*1000)
#endif

/************************************************************************************************
  在OpenNIGrabber采集循环执行的回调函数cloud_cb_中，首先把获取的点云压缩到stringstream缓冲区，下一步就是解压缩，
  它对压缩了的二进制数据进行解码，存储在新的点云中解码了点云被发送到点云可视化对象中进行实时可视化
*************************************************************************************************/

  
class SimpleOpenNIViewer
{
public:
  // 类构造函数 
  SimpleOpenNIViewer () :
    viewer (" Point Cloud Compression Example")//基于 可视化对象创建
  {}

  void
  cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
  {
    if (!viewer.wasStopped ())
    {
      // stringstream to store compressed point cloud
      std::stringstream compressedData;//压缩后的点云 存储压缩点云的字节流对象
      // 存储输出点云
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut (new pcl::PointCloud<pcl::PointXYZRGBA> ());

      // compress point cloud  点云压缩编码  压缩到stringstream缓冲区
      PointCloudEncoder->encodePointCloud (cloud, compressedData);

      // decompress point cloud 点云解码 输出
      PointCloudDecoder->decodePointCloud (compressedData, cloudOut);


      // show decompressed point cloud
      viewer.showCloud (cloudOut);//可视化
    }
  }
/**************************************************************************************************************
 在函数中创建PointCloudCompression类的对象来编码和解码，这些对象把压缩配置文件作为配置压缩算法的参数
 所提供的压缩配置文件为OpenNI兼容设备采集到的点云预先确定的通用参数集，本例中使用MED_RES_ONLINE_COMPRESSION_WITH_COLOR
 配置参数集，用于快速在线的压缩，压缩配置方法可以在文件/io/include/pcl/compression/compression_profiles.h中找到，
  在PointCloudCompression构造函数中使用MANUAL——CONFIGURATION属性就可以手动的配置压缩算法的全部参数
******************************************************************************************/
  void
  run ()
  {

    bool showStatistics = true;//设置在标准设备上输出打印出压缩结果信息

    //  压缩选项详见 /io/include/pcl/compression/compression_profiles.h  分辨率5mm3，有颜色，快速在线编码
    // http://www.pclcn.org/study/shownews.php?lang=cn&id=125
    pcl::io::compression_Profiles_e compressionProfile = pcl::io::MED_RES_ONLINE_COMPRESSION_WITH_COLOR;

    // 初始化压缩与解压缩对象，其中压缩对象需要设定压缩参数选项，解压缩按照数据源自行判断
    PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA> (compressionProfile, showStatistics);
    PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA> ();

   /***********************************************************************************************************
    下面的代码为OpenNI兼容设备实例化一个新的采样器，并且启动循环回调接口，每从设备获取一帧数据就回调函数一次，，这里的
    回调函数就是实现数据压缩和可视化解压缩结果。
   ************************************************************************************************************/

    // 创建从OpenNI获取点云的抓取对象
    pcl::Grabber* interface = new pcl::OpenNIGrabber ();

    //创建从 OpenNI获取点云的抓取对象  这里的回调函数实现数据压缩和可视化解压缩结果。
    boost::function<void   // 建立回调函数
    (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

    // 建立回调函数与回调信号之间绑定
    boost::signals2::connection c = interface->registerCallback (f);

    // 开始接收点云数据流
    interface->start ();

    while (!viewer.wasStopped ())
    {
      sleep (1);
    }

    interface->stop ();

     // 删除压缩与解压缩的实例
    delete (PointCloudEncoder);
    delete (PointCloudDecoder);

  }

  pcl::visualization::CloudViewer viewer;
// 创建PointCloudCompression类的对象来编码和解码
  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudEncoder;
  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudDecoder;

};

int
main (int argc, char **argv)
{
  SimpleOpenNIViewer v;//创建一个新的SimpleOpenNIViewer  实例并调用他的run方法
  v.run ();

  return (0);
}
/*
压缩选项详见 /io/include/pcl/compression/compression_profiles.h

LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR:分辨率1cm3，无颜色，快速在线编码

LOW_RES_ONLINE_COMPRESSION_WITH_COLOR:分辨率1cm3，有颜色，快速在线编码

MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR:分辨率5mm3，无颜色，快速在线编码

MED_RES_ONLINE_COMPRESSION_WITH_COLOR:分辨率5mm3，有颜色，快速在线编码

HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR:分辨率1mm3，无颜色，快速在线编码

HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR:分辨率1mm3，有颜色，快速在线编码

LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR:分辨率1cm3，无颜色，高效离线编码

LOW_RES_OFFLINE_COMPRESSION_WITH_COLOR:分辨率1cm3，有颜色，高效离线编码

MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR:分辨率5mm3，无颜色，高效离线编码

MED_RES_OFFLINE_COMPRESSION_WITH_COLOR:分辨率5mm3，有颜色，高效离线编码

HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR:分辨率5mm3，无颜色，高效离线编码

HIGH_RES_OFFLINE_COMPRESSION_WITH_COLOR:分辨率5mm3，有颜色，高效离线编码

MANUAL_CONFIGURATION允许为高级参数化进行手工配置

*/
