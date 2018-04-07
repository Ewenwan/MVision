/*
点云坐标变换
变换矩阵的其次表示
[R t]
绕Z轴旋转 45度 （顺时针） 沿着x轴平移 2.5个单位

x' = x * cos(cet) + y * sin(cet) + 0 * z
y' = y * cos(cet) - x * sin(cet) + 0 *z
z' = 0*x +0*y + 1*z
所以旋转矩阵为
cos(cet) -sin(cet) 0
sin(cet)  cos(cet) 0
0           0      1
平移矩阵为
2.5
0
0

点云数据可以用ASCII码的形式存储在PCD文件中（
关于该格式的描述可以参考链接：The PCD (Point Cloud Data) file format）。
为了生成三维点云数据，在excel中用rand()函数生成200行0-1的小数，ABC三列分别代表空间点的xyz坐标。

PDC文件格式
# .PCD v.7 - Point Cloud Data file format
VERSION .7        
FIELDS x y z     
SIZE 4 4 4         
TYPE F F F         
COUNT 1 1 1     
WIDTH 200        
HEIGHT 1        
VIEWPOINT 0 0 0 1 0 0 0
POINTS 200        
DATA ascii        
0.88071666    0.369209703    0.062937221
0.06418104    0.579762553    0.221359779
...
...
0.640053058    0.480279041    0.843647334
0.245554712    0.825770496    0.626442137

内容格式
FIELDS x y z                                # XYZ data
FIELDS x y z rgb                            # XYZ + colors
FIELDS x y z normal_x normal_y normal_z     # XYZ + surface normals
FIELDS j1 j2 j3                             # moment invariants
存储大小
SIZE - specifies the size of each dimension in bytes. Examples:

    unsigned char/char has 1 byte
    unsigned short/short has 2 bytes
    unsigned int/int/float has 4 bytes
    double has 8 bytes
数据类型
TYPE - specifies the type of each dimension as a char. The current accepted types are:

    I - represents signed types int8 (char), int16 (short), and int32 (int)
    U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
    F - represents float types

视角
The viewpoint information is specified as a translation (tx ty tz) + quaternion (qw qx qy qz). The default value is:
VIEWPOINT 0 0 0 1 0 0 0



*/
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>                  //allows us to use pcl::transformPointCloud function
#include <pcl/visualization/pcl_visualizer.h>
#include <ctime>//time

// This is the main function
int main (int argc, char** argv)
{
    srand (time (NULL));//随机数

    // 点云 creates a PointCloud<PointXYZ> boost shared pointer and initializes it.
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>& source_cloud = *source_cloud_ptr;


    // Load PCD file
    //if (pcl::io::loadPCDFile<pcl::PointXYZ> ("sample.pcd", *source_cloud) == -1) 
    //{
    //    PCL_ERROR ("Couldn't read file sample.pcd \n");
    //    return (-1);
    //}

  // 产生假的点云数据
  source_cloud.width = 500;//500数据点
  source_cloud.height = 1;
  source_cloud.points.resize (source_cloud.width * source_cloud.height);

  for (size_t i = 0; i < source_cloud.points.size (); ++i)
  {
   // 0~1之间的数
    source_cloud.points[i].x = 1.0f * rand () / (RAND_MAX + 1.0f);
    source_cloud.points[i].y = 1.0f * rand () / (RAND_MAX + 1.0f);
    source_cloud.points[i].z = 1.0f * rand () / (RAND_MAX + 1.0f);
  }

    /* 
    变换矩阵 
    Reminder: how transformation matrices work :

           |-------> This column is the translation
    | 1 0 0 x |  \
    | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
    | 0 0 1 z |  /
    | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

    METHOD #1: Using a Matrix4f
    This is the "manual" method, perfect to understand but error prone !
    */
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

    // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    // Here we defined a 45° (PI/4) rotation around the Z axis and a translation on the X axis.
    float theta = M_PI/4; // The angle of rotation in radians
    transform_1 (0,0) = cos (theta);
    transform_1 (0,1) = -sin(theta);
    transform_1 (1,0) = sin (theta);
    transform_1 (1,1) = cos (theta);
    //    (row, column)

    // Define a translation of 2.5 meters on the x axis.
    transform_1 (0,3) = 2.5;

    // Print the transformation
    printf ("Method #1: using a Matrix4f\n");
    std::cout << transform_1 << std::endl;
    // 执行转换 Executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>& transformed_cloud = *transformed_cloud_ptr;
    /*
    void pcl::transformPointCloud(const pcl::PointCloud< PointT > & cloud_in, 
                                    pcl::PointCloud< PointT > &  cloud_out,  
                                    const Eigen::Matrix4f &  transform  ) 
    */
    // Apply an affine transform defined by an Eigen Transform.
    pcl::transformPointCloud (source_cloud, transformed_cloud, transform_1);

    // 可视化 Visualization
    std::cout << "\nPoint cloud colors :  white  = original point cloud\n"
              << "red  = transformed point cloud\n"<< std::endl;
    pcl::visualization::PCLVisualizer viewer ("point_Matrix_transformation");
    // ===================================
    // 源点云显示颜色 白色 Define R,G,B colors for the point cloud==
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud_ptr, 255, 255, 255);
    // We add the point cloud to the viewer and pass the color handler
    viewer.addPointCloud (source_cloud_ptr, source_cloud_color_handler, "original_cloud");

    // 转换后的点云 红色================
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud_ptr, 230, 20, 20); // Red
    viewer.addPointCloud (transformed_cloud_ptr, transformed_cloud_color_handler, "transformed_cloud");

//=========显示坐标系 建立空间直角坐标系 ======
    viewer.addCoordinateSystem (1.0, 0);  //Adds 3D axes describing a coordinate system to screen at 0,0,0. 
//=========背景颜色=======
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
//=========
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    //viewer.setPosition(800, 400); // 可视化窗口大小 Setting visualiser window position

    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();//按q键停止显示 退出
    }

  return 0;
}
