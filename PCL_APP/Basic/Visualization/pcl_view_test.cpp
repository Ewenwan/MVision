// 载入点云 滤除不的点云 读取识别结果 显示物体三维框
/*
// CMakeLists.txt

#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( global_view )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

 # 矩阵 Eigen3
find_package(Eigen3 3.1.0 REQUIRED)

#找 pcl 并链接 
#注意　common io filters visualization　分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io filters visualization)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS} ${EIGEN3_INCLUDE_DIR})
add_definitions( ${PCL_DEFINITIONS} )

# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

#可执行文件 直通滤波器 PassThrough　
add_executable( view view.cpp )
target_link_libraries( view ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_LIBS} 
*/

/*
// result4.txt
3d tvmonitor 0.72718 -0.41 -0.27 1.65 -0.27 -0.11 1.68 -0.68 -0.41 1.60
3d chair 0.75304 0.09 0.05 2.00 0.27 0.39 2.05 -0.16 -0.25 1.85
3d tvmonitor 0.52181 0.41 0.49 1.85 0.61 0.63 1.89 0.22 0.35 1.79
3d chair 0.67358 0.80 0.04 1.98 0.98 0.31 2.05 0.62 -0.25 1.89
*/

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>//直通滤波器 PassThrough　
#include <pcl/visualization/cloud_viewer.h>//点云可视化
#include <pcl/visualization/pcl_visualizer.h> // PCLVisualizer 点云可视化器
#include <pcl/io/pcd_io.h>//点云文件pcd 读写

#include<sstream>// 字符串

#include<string>// 字符串
#include<map>// 

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// 别名
typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point>  Cloud;


/////////////////==== 数据标识映射====///////////////////
// voc 类别:id 映射 
std::map<std::string, int>::value_type init_value[] =
{
    std::map<std::string, int>::value_type( "background", 0),
    std::map<std::string, int>::value_type( "aeroplane", 1),
    std::map<std::string, int>::value_type( "bicycle", 2),
    std::map<std::string, int>::value_type( "bird", 3),
    std::map<std::string, int>::value_type( "boat", 4),
    std::map<std::string, int>::value_type( "bottle", 5),
    std::map<std::string, int>::value_type( "bus", 6),
    std::map<std::string, int>::value_type( "car", 7),
    std::map<std::string, int>::value_type( "cat", 8),
    std::map<std::string, int>::value_type( "chair", 9),
    std::map<std::string, int>::value_type( "cow", 10),
    std::map<std::string, int>::value_type( "diningtable", 11),
    std::map<std::string, int>::value_type( "dog", 12),
    std::map<std::string, int>::value_type( "horse", 13),
    std::map<std::string, int>::value_type( "motorbike", 14),
    std::map<std::string, int>::value_type( "person", 15),
    std::map<std::string, int>::value_type( "pottedplant", 16),
    std::map<std::string, int>::value_type( "sheep", 17),
    std::map<std::string, int>::value_type( "sofa", 18),
    std::map<std::string, int>::value_type( "train", 19),
    std::map<std::string, int>::value_type( "tvmonitor", 20),
};
 static std::map<std::string, int> obj_class_map(init_value, init_value+21); //


// 目标语义信息
typedef struct Cluster
{
 std::string object_name; // 物体类别名
 int class_id;            // 对应类别id
 float prob;              // 置信度
 Eigen::Vector3f minPt;   // 所有点中最小的x值，y值，z值
 Eigen::Vector3f maxPt;   // 所有点中最大的x值，y值，z值
 Eigen::Vector3f centroid;// 点云中心点 
} Cluster;


void add_cube(pcl::visualization::PCLVisualizer &  viewer, 
                                 std::vector<Cluster> & clusters, 
                                 std::vector<cv::Scalar> & colors)
{
    viewer.removeAllShapes();// 去除 之前 已经显示的形状
    for(int i=0; i<clusters.size(); i++)
    {
        Cluster& cluster  = clusters[i];
        std::string& name = cluster.object_name;  // 物体类别名
        int&   class_id   = cluster.class_id;     // 类别id
        float& prob = cluster.prob;               // 置信度
	Eigen::Vector3f& minPt = cluster.minPt;   // 所有点中最小的x值，y值，z值
	Eigen::Vector3f& maxPt = cluster.maxPt;   // 所有点中最大的x值，y值，z值
	Eigen::Vector3f& centr = cluster.centroid;// 点云中心点
        Eigen::Vector3f boxCe = (maxPt + minPt)*0.5f; // 盒子中心
        Eigen::Vector3f boxSi = maxPt - minPt;        // 盒子尺寸
          
	//fprintf(stderr, "3d %s %.5f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
	//	        name.c_str(), prob, centr[0], centr[1], centr[2], 
         //               maxPt[0], maxPt[1], maxPt[2], minPt[0], minPt[1], minPt[2]);
               // 打印名字、置信度、中心点坐标

	const Eigen::Quaternionf quat(Eigen::Quaternionf::Identity());// 姿态 四元素
	std::string name_new = name + boost::chrono::to_string(i);    // 包围框的名字
	viewer.addCube(boxCe, quat, boxSi[0], boxSi[1], boxSi[2], name_new.c_str()); // 添加盒子
	viewer.setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
	                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 
			name_new.c_str());
	viewer.setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_COLOR,
                        colors[class_id].val[2]/255.0, 
                        colors[class_id].val[1]/255.0, colors[class_id].val[0]/255.0,// 颜色
			name_new.c_str());//
    }
} 

int main (int argc, char** argv)
{
  // 定义　点云对象　指针
   Cloud::Ptr cloud_ptr(new Cloud());
   Cloud::Ptr cloud_filtered_ptr(new Cloud());
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);

  // 读取点云文件　填充点云对象
  pcl::PCDReader reader;
  reader.read( "../global_color.pcd", *cloud_ptr);
  if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  std::cerr << "Cloud before filtering滤波前: " << cloud_ptr->points.size() << std::endl;

  // 载入坐标等数据
    // 定义输入文件流类对象infile
    ifstream infile("../result4.txt",ios::in);
 
    if(!infile){  // 判断文件是否存在
      cerr<<"open error."<<endl;
      exit(1); // 退出程序
    }
 
    char str[255]; // 定义字符数组用来接受读取一行的数据
    std::string tep;
    std::vector<std::vector<std::string>> obj;// 二维数值
    while(infile)
    {
        infile.getline(str,255);  // getline函数可以读取整行并保存在str数组里
        //std::cout << str << std::endl;
        std::stringstream input(str);
        std::vector<std::string> res;
        while(input>>tep)
            res.push_back(tep);
        obj.push_back(res); 
    }
    
    std::vector<cv::Scalar> colors;
// 物体颜色
    for (int i = 0; i < 21; i++) // 带有背景
    { 
       colors.push_back(cv::Scalar(i*10 + 40, i*10 + 40, i*10 + 40));
    }
    colors[5] = cv::Scalar(255,0,255); // 瓶子 粉色    bgr
    colors[9] = cv::Scalar(255,0,0);   // 椅子 蓝色
    colors[15] = cv::Scalar(0,0,255);  // 人 红色
    colors[20] = cv::Scalar(0,255,0);  // 显示器 绿色 

    std::vector<Cluster> clusters;
    std::cout << obj.size() << std::endl; // 为啥会是5?
    for(int i=0; i<obj.size()-1; i++)
    {
        std::vector<std::string> oj = obj[i];// 一个目标的信息
        // 3d tvmonitor 0.72718 -0.41 -0.27 1.65 -0.27 -0.11 1.68 -0.68 -0.41 1.60
        //      种类     概率    中心点          |    最大        |  最小  x,y,z
        
        Cluster cluster;
        cluster.object_name = oj[1];
        cluster.class_id    = obj_class_map[oj[1]];
        cluster.prob        = atof(oj[2].c_str());
        cluster.minPt       = Eigen::Vector3f(atof(oj[9].c_str()), atof(oj[10].c_str()), atof(oj[11].c_str()));
        cluster.maxPt       = Eigen::Vector3f(atof(oj[6].c_str()), atof(oj[7].c_str()),  atof(oj[8].c_str()));
        cluster.centroid    = Eigen::Vector3f(atof(oj[3].c_str()), atof(oj[4].c_str()),  atof(oj[5].c_str()));;

        clusters.push_back(cluster);

        fprintf(stderr, "3d %d %s %.5f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
		        cluster.class_id, cluster.object_name.c_str(), cluster.prob, 
                        cluster.centroid[0], cluster.centroid[1], cluster.centroid[2], 
                        cluster.maxPt[0], cluster.maxPt[1], cluster.maxPt[2], 
                        cluster.minPt[0], cluster.minPt[1], cluster.minPt[2]);   
    }


  // 创建滤波器对象　Create the filtering object
  pcl::PassThrough<Point> pass;
  pass.setInputCloud (cloud_ptr);//设置输入点云
  pass.setFilterFieldName ("z");// 定义轴
  pass.setFilterLimits (0.0, 2.1);//　范围
  pass.setFilterLimitsNegative (false);//标志为false时保留范围内的点, true时保留发内外的点
  pass.filter (*cloud_filtered_ptr);

  // 输出滤波后的点云
  std::cerr << "Cloud after filtering滤波后: " << cloud_filtered_ptr->points.size() <<  std::endl;

  // 程序可视化
  //pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  //viewer.showCloud(cloud_filtered_ptr);
  //while (!viewer.wasStopped())
  //  {
        // Do nothing but wait.
  //  }

  pcl::visualization::PCLVisualizer viewer;
  add_cube(viewer, clusters, colors);//
  pcl::visualization::PointCloudColorHandlerRGBField<Point> rgb(cloud_filtered_ptr);
  viewer.removePointCloud("SemMap");// 去除原来的点云====
  viewer.addPointCloud<Point> (cloud_filtered_ptr, rgb, "SemMap");
  viewer.setPointCloudRenderingProperties (
                       pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "SemMap");

  while(!viewer.wasStopped ())
  {
   //obj_pt = cluster_large.c_ptr;
    viewer.spinOnce (100);// 
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));// 
   //usleep(3000); 
  }

  return (0);
}
