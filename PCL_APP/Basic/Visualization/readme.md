# 可视化

    可视化（visualization）是利用计算机图形学和图像处理技术，
    将数据转换图像在屏幕上显示出来，并进行交互处理的的理论，方法和技术.

    [参考](https://blog.csdn.net/u013019296/article/details/70052309)

## 命令行显示 pcl_viewer

    linux 下可直接在命令行输入 pcl_viewr path/to/.pcd或.vtk可直接显示pcl中的点云文件。

    pcl_viewr几个常用的命令：
    r键: 重现视角。如果读入文件没有在主窗口显示，不妨按下键盘的r键一试。
    j键：截图功能。
    g键：显示/隐藏 坐标轴。
    鼠标：左键，使图像绕自身旋转; 滚轮, 按住滚轮不松，可移动图像，滚动滚轮，可放大/缩小 图像; 右键,“原地” 放大/缩小。
    -/+：-（减号）可缩小点; +(加号)，可放大点。
    pcl_viewe -bc r,g,b /path/to/.pcd:可改变背景色.
    pcl_viewer还可以用来直接显示pfh，fpfh（fast point feature histogram），vfh等直方图。
    常用的pcl_viewer 好像就这些，其他未涉及到的功能可通过pcl_viewer /path/.pcd 打开图像，按键盘h（获取帮助）的方式获得.

## pcl_visualization库建立了能够快速建立原型的目的和可视化算法对三维点云数据操作的结果。
    类似于opencv的highgui例程显示二维图像，在屏幕上绘制基本的二维图形，库提供了以下几点：

      （1）渲染和设置视觉特性的方法（如颜色、大小、透明度等）在PCL任意n维的点云数据集 pcl::PointCloud<T> format
      （2）在屏幕上绘制基本的3D形状的方法（例如，圆柱体，球体，线，多边形等），无论是从点集或参数方程；
      （3）一个直方图可视化模块（pclhistogramvisualizer）的二维图；
      （4）大量的几何和颜色处理pcl::PointCloud<T> datasets
      （5）a pcl::RangeImage 范围图像 深度图像 可视化模块




## "复杂的"可视化类  pcl::visualization::PCLVisualizer
    #####
    键盘交互
        #include <pcl/visualization/pcl_visualizer.h> //包含基本可视化类
        #include <pcl/visualization/pcl_visualizer.h>
        #include <boost/thread/thread.hpp>
        //设置键盘交互函数,按下`space`键，某事发生
        void keyboardEvent(const pcl::visualization::KeyboardEvent &event,void *nothing)
        {
            if(event.getKeySym() == "space" && event.keyDown())
                    next_iteration = true;
        }
        int main (int argc, char **argv)
        {
          //  1.  读入点云 source, target    
          // 2.  处理读入的数据文件

      boost::shared_ptr<pcl::visualization::PCLVisualizer> view (new pcl::visualization::PCLVisualizer("test")); //创建可视化窗口，名字叫做`test`
           view->setBackgroundColor(0.0,0,0); //设置背景色为黑色
           viewer->addCoordinateSystem(1.0); //建立空间直角坐标系
          //viewer->setCameraPosition(0,0,200); //设置坐标原点
           viewer->initCameraParameters();   //初始化相机参数

          //***`*显示的”处理的数据文件“的具体内容*`***
              view->registerKeyboardCallback(&keyboardEvent,(void*)NULL);  //设置键盘回吊函数
              while(!viewer->wasStopped())
              {
                viewer->spinOnce(100);  //显示
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));   //随时间
              }

        }


    ####################################################
##  在pcl中, 有一类可以画两点之间线段的函数，绘制点之间连线的方法十分有用，
    例如，显示两组点云之间的对应点关系时，可方便用户直观的观看点云之间的对应关系。
    它是可视化函数pcl::visualizeton的一员。具体用法如下：

    #include <iostream>
    #include <pcl/visualization/pcl_visualizer.h>
    #include <pcl/point_types.h>
    #include <boost/thread/thread.hpp>
    using namespace std;    
    typedef pcl::PointCloud<pcl::PointXYZ>  pointcloud;
    int main(int argc, char *argv[])
    {
        pointcloud::Ptr  cloud (new pointcloud);
        pcl::PointXYZ a, b, z;
        a.x = 0;
        a.y = 0;
        a.z = 0;
        b.x  = 5;
        b.y  = 8;
        b.z = 10;
        z.x = 4;
        z.y = 3;
        z.z = 20;

          boost::shared_ptr<pcl::visualization::PCLVisualizer> view (new pcl::visualization::PCLVisualizer ("line Viewer"));
          view->setBackgroundColor(r,g,b); //背景色
          view->addLine<pcl::PointXYZ>(a,b,"line");
        view->addLine<pcl::PointXYZ> (a,z,255,,0,0,"line1"); //红色线段,线的名字叫做"line1"
        view->addArrow<pcl::PointXYZ> (b,z,255,0,0,"arrow");  //带箭头
           while (!view->wasStopped ())
           {
               view->spinOnce(100);
               boost::this_thread::sleep (boost::posix_time::microseconds (100000));

           }

        return 0;
    }
    
    这个小程序主要是画了两条线段，a-b,名字叫做“line”; 
    a-z. 名字叫做"line1"其中a-z为红色，
    在addLine<PointT>函数中，其原型为addLine(p1,p2,r,g,b, "viewport");
    addArrow(b,z,r,g,b,"id name"), 
    主要是画出从b指向z的带箭头的线段，其颜色由用户指定，也可默认。


