# pcl::getMinMax3D函数
                
## 1)查看API说明：

    void pcl::getMinMax3D 	( 	const pcl::PointCloud< PointT > &  	cloud,
        const std::vector< int > &  	indices,
        Eigen::Vector4f &  	min_pt,
        Eigen::Vector4f &  	max_pt	 
      ) 	
    Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud.

## Parameters:
 
    cloud  the point cloud data message

    indices  the vector of point indices to use from cloud

    min_pt  the resultant minimum bounds

    max_pt  the resultant maximum bounds

    即cloud为输入点云，而非指针（共享指针则写为*cloud），
    输出min_pt为所有点中最小的x值，y值，z值，
    输出max_pt为为所有点中最大的x值，y值，z值。

## (2)程序验证
```c
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
int
main (int, char**)
{
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  // cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/exbot/文档/pp/pcd/123.pcd", *cloud);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width=5;
  cloud.height=1;
  cloud.is_dense=false;
  cloud.points.resize(cloud.width*cloud.height); 
  cloud.points[0].x=0.0f;
  cloud.points[0].y=0.0f;
  cloud.points[0].z=0.0f; 
  cloud.points[1].x=-1.0f;
  cloud.points[1].y=0.0f;
  cloud.points[1].z=0.0f;
  cloud.points[2].x=0.0f;   
  cloud.points[2].y=-1.0f;
  cloud.points[2].z=0.0f; 
  cloud.points[3].x=0.0f;
  cloud.points[3].y=0.0f;
  cloud.points[3].z=2.0f;
  cloud.points[4].x=1.0f;
  cloud.points[4].y=1.0f;
  cloud.points[4].z=1.0f;  
  pcl::PointXYZ minPt, maxPt;
 // pcl::getMinMax3D (*cloud, minPt, maxPt);
  pcl::getMinMax3D (cloud, minPt, maxPt);
  std::cout << "Max x: " << maxPt.x << std::endl;
  std::cout << "Max y: " << maxPt.y << std::endl;
  std::cout << "Max z: " << maxPt.z << std::endl;
  std::cout << "Min x: " << minPt.x << std::endl;
  std::cout << "Min y: " << minPt.y << std::endl;
  std::cout << "Min z: " << minPt.z << std::endl;
  return (0);
}
```

## 编译运行输出为：
    Max x: 1
    Max y: 1
    Max z: 2
    Min x: -1
    Min y: -1
    Min z: 0所有点中最小值，而非某个点。
