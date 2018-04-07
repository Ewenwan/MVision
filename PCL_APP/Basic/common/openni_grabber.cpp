#include <pcl/point_cloud.h>         //点云类定义头文件
#include <pcl/point_types.h>        //点 类型定义头文件
#include <pcl/io/openni_grabber.h>  //OpenNI数据流获取头文件
#include <pcl/common/time.h>        //时间头文件


//类SimpleOpenNIProcessor  的回调函数，作为在获取数据时，
//对数据进行处理的回调函数的封装，在本例中并没有什么处理，只是实时的在标准输出设备打印处信息。
class SimpleOpenNIProcessor
{
public:
  void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
  {
    static unsigned count = 0;
    static double last = pcl::getTime ();  //获取当前时间
    if (++count == 30)                     //每30ms一次输出
    {
      double now = pcl::getTime ();
      //  >> 右移
      std::cout << "distance of center pixel :" << cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z <<
                " mm. Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
      count = 0;
      last = now;
    }
  }
  
  void run ()
  {
    pcl::Grabber* interface = new pcl::OpenNIGrabber();  //创建OpenNI采集对象

    // 定义回调函数
    boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
      boost::bind (&SimpleOpenNIProcessor::cloud_cb_, this, _1);
    boost::signals2::connection c = interface->registerCallback (f);//注册回调函数
    interface->start ();   //开始接受点云数据
     //直到用户按下Ctrl -c
    while (true)
      boost::this_thread::sleep (boost::posix_time::seconds (1));
    //   停止采集
    interface->stop ();
  }
};

int main ()
{
  SimpleOpenNIProcessor v;
  v.run ();
  return (0);
}
