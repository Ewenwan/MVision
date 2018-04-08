/*
Range Images范围图像，意思是通过点云获得特定角度的观察图像.
以特定角度投影点云到二维平面上，根据距离设定像素值，显示图像。

在3D视窗中以点云形式进行可视化（深度图像来自于点云），
另一种是将深度值映射为颜色，从而以彩色图像方式可视化深度图像， 

深度图 到 点云
点云 范围图像 到 深度图

怎样可视化深度图像

本小节讲解如何可视化深度图像的两种方法，
在3D视窗中以点云形式进行可视化（深度图像来源于点云），
另一种是，将深度值映射为颜色，从而以彩色图像方式可视化深度图像。

学习如何从点云和给定的传感器位置来创建深度图像，
下面的程序，首先是生成一个矩形点云，然后基于该点云创建深度图像。
在3D视窗中以点云形式进行可视化（深度图像来自于点云），另一种是将深度值映射为颜色，从而以彩色图像方式可视化深度图像， 

*/
#include <pcl/range_image/range_image.h>//深度图像头文件

int main (int argc, char** argv) {
// 点云对象
  pcl::PointCloud<pcl::PointXYZ> pointCloud;
  
// 产生数据 生成一个矩形点云
  for (float y=-0.5f; y<=0.5f; y+=0.01f) {
    for (float z=-0.5f; z<=0.5f; z+=0.01f) {
      pcl::PointXYZ point;//单个点
      point.x = 2.0f - y;
      point.y = y;
      point.z = z;
      pointCloud.points.push_back(point);//循环添加点数据到点云对象
    }
  }
  pointCloud.width = (uint32_t) pointCloud.points.size();
  pointCloud.height = 1;//设置点云对象的头信息
  
// 设置参数 We now want to create a range image from the above point cloud, with a 1deg angular resolution
/*
这部分定义了创建深度图像时需要的设置参数，将角度分辨率定义为1度，
意味着由邻近的像素点所对应的每个光束之间相差1度，
maxAngleWidth=360和maxAngleHeight=180意味着，
我们进行模拟的距离传感器对周围的环境拥有一个完整的360度视角，
用户在任何数据集下都可以使用此设置，
因为最终获取的深度图像将被裁剪到有空间物体存在的区域范围。

*/
  float angularResolution = (float) (  1.0f * (M_PI/180.0f)); // 按弧度1度
  float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f)); // 按弧度360.0度
  float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f)); // 按弧度180.0度
// ensorPose定义了模拟深度图像获取传感器的6自由度位置，其原始值为横滚角roll、俯仰角pitch、偏航角yaw都为0
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);//采集位置
// coordinate_frame=CAMERA_FRAME说明系统的X轴是向右的，Y轴是向下的，Z轴是向前的(右手坐标系 拇指为z轴，食指为x轴)，
// 另外一个选择是LASER_FRAME，其X轴向前，Y轴向左，Z轴向上。
  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;//深度图像遵循的坐标系统
//noiseLevel=0是指使用一个归一化的Z缓冲器来创建深度图像，
//但是如果想让邻近点集都落在同一个像素单元，用户可以设置一个较高的值，
//例如noiseLevel=0.05可以理解为，深度距离值是通过查询点半径为5cm的圆内包含的点用来平均计算而得到的。
  float noiseLevel=0.00;
//如果minRange>0，则所有模拟器所在位置半径minRange内的邻近点都将被忽略，即为盲区。
  float minRange = 0.0f;
//在裁剪图像时，如果borderSize>0，将在图像周围留下当前视点不可见点的边界。
  int borderSize = 1;
  
  pcl::RangeImage rangeImage;//范围图像 深度图像
  rangeImage.createFromPointCloud(pointCloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
  
  std::cout << rangeImage << "\n";
}
