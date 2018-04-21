/*********************************************************************
* Software License Agreement (BSD License)
* 打开图片/打开设备/打开视频 发布相机信息流
*********************************************************************/

#include <ros/ros.h>
#include <nodelet/nodelet.h>
//使用ROS中nodelet包可以实现在同一个进程内同时运行多种算法，且算法之间通信开销零拷贝。
#include <cv_bridge/cv_bridge.h>// opencv 格式　转换到 ros格式
#include <image_publisher/ImagePublisherConfig.h>//动态参数配置文件
#include <image_transport/image_transport.h>// 发布jpg图片文件流
#include <sensor_msgs/CameraInfo.h>//相机信息
#include <camera_info_manager/camera_info_manager.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dynamic_reconfigure/server.h>//动态参数配置服务器
#include <boost/assign.hpp>
using namespace boost::assign;//(标准容器填充库)
// 自定义　图像发布　命名空间
namespace image_publisher {
class ImagePublisherNodelet : public nodelet::Nodelet
{
  dynamic_reconfigure::Server<image_publisher::ImagePublisherConfig> srv;

  image_transport::CameraPublisher pub_;//相机图片　发布

  boost::shared_ptr<image_transport::ImageTransport> it_;//　照片传输类指针
  ros::NodeHandle nh_;//

  cv::VideoCapture cap_;// 相机捕捉类
  cv::Mat image_;//图像
  int subscriber_count_;
  ros::Timer timer_;//时间

  std::string frame_id_;
  std::string filename_;
  bool flip_image_;
  int flip_value_;
  sensor_msgs::CameraInfo camera_info_;//相机信息
  
  // 相机动态参数配置　回调函数　输入动态参数配置类（cfg文件生成）
  void reconfigureCallback(image_publisher::ImagePublisherConfig &new_config, uint32_t level)
  {
    frame_id_ = new_config.frame_id;//坐标系
  // 按照发布频率计算等待休眠的时间间隔　  来执行　任务函数　do_work
    timer_ = nh_.createTimer(ros::Duration(1.0/new_config.publish_rate), &ImagePublisherNodelet::do_work, this);

    camera_info_manager::CameraInfoManager c(nh_);// 相机信息管理类
    if ( !new_config.camera_info_url.empty() ) {//　参数传递了　相机信息url　
      try {
        c.validateURL(new_config.camera_info_url);//检测　验证有效性
        c.loadCameraInfo(new_config.camera_info_url);//载入相机参数
        camera_info_ = c.getCameraInfo();//　类内相机信息类初始化
      } catch(cv::Exception &e) {
        NODELET_ERROR("camera calibration failed to load: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
      }
    }
  }

  void do_work(const ros::TimerEvent& event)
  {
    // Transform the image.
    try
    {
      if ( cap_.isOpened() ) {//打开
        if ( ! cap_.read(image_) ) {
          cap_.set(CV_CAP_PROP_POS_FRAMES, 0);//视频文件　第0帧
        }
      }
      if (flip_image_)//矩阵(图像)镜像(翻转)　（水平　垂直　　水平+垂直）
        cv::flip(image_, image_, flip_value_);
      // opencv 图像格式　转换到　ros　相机传感器消息类型
      sensor_msgs::ImagePtr out_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_).toImageMsg();//传感器信息　rgb８位
      out_img->header.frame_id = frame_id_;//图像信息　坐标系
      out_img->header.stamp = ros::Time::now();//时间戳
      camera_info_.header.frame_id = out_img->header.frame_id;//相机信息　坐标系
      camera_info_.header.stamp = out_img->header.stamp;//时间戳

      pub_.publish(*out_img, camera_info_);//发布信息
    }
    catch (cv::Exception &e)
    {
      NODELET_ERROR("Image processing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
    }
  }

  void connectCb(const image_transport::SingleSubscriberPublisher& ssp)
  {
    subscriber_count_++;
  }

  void disconnectCb(const image_transport::SingleSubscriberPublisher&)
  {
    subscriber_count_--;
  }

public:
  virtual void onInit()//类初始化会执行的函数
  {
    subscriber_count_ = 0;
    nh_ = getPrivateNodeHandle();//私有节点
    it_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh_));
    pub_ = image_transport::ImageTransport(nh_).advertiseCamera("image_raw", 1);//图像传输类　发布相机信息话题　到　image_raw 上

    nh_.param("filename", filename_, std::string(""));//获取　（图片/视频/相机设备号）　名称
    NODELET_INFO("File name for publishing image is : %s", filename_.c_str());
    //image_ = cv::imread(filename_,  CV_LOAD_IMAGE_COLOR);//发布的图片  这里有问题？？？？
    image_ = cv::imread(filename_.c_str(),  cv::IMREAD_COLOR);
    if(image_.empty()) std::cout << "obtain image error" << std::endl;
///*
    try {
      image_ = cv::imread(filename_, CV_LOAD_IMAGE_COLOR);//发布的图片
      if ( image_.empty() ) { // if filename is motion file or device file
        try {  // if filename is number　　打开相机设备
          //int num = boost::lexical_cast<int>(filename_.c_str());//num is 1234798797
          int num = 0;
          std::cout << "canmre id : " << num << std::endl;  
          //cap_.open(num);
          if(!cap_.isOpened()) {std::cout << "camera open err" << std::endl; return;}
        } catch(boost::bad_lexical_cast &) { // if file name is string
          cap_.open(filename_);//视频文件
	  if(!cap_.isOpened()) std::cout << "obtain image error" << std::endl;
        }
        CV_Assert(cap_.isOpened());//打开图片/打开设备/打开视频完成
        cap_.read(image_);// 图片文件
        cap_.set(CV_CAP_PROP_POS_FRAMES, 0);
      }
      CV_Assert(!image_.empty());
    }
    catch (cv::Exception &e)
    {
      NODELET_ERROR("Failed to load image (%s): %s %s %s %i", filename_.c_str(), e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
    }
//*/
///*
    bool flip_horizontal;//　水平镜像
    nh_.param("flip_horizontal", flip_horizontal, false);
    NODELET_INFO("Flip horizontal image is : %s",  ((flip_horizontal)?"true":"false"));

    bool flip_vertical;//　垂直镜像
    nh_.param("flip_vertical", flip_vertical, false);
    NODELET_INFO("Flip flip_vertical image is : %s", ((flip_vertical)?"true":"false"));

    // From http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#void flip(InputArray src, OutputArray dst, int flipCode)
    // FLIP_HORIZONTAL == 1, FLIP_VERTICAL == 0 or FLIP_BOTH == -1
    flip_image_ = true;
    if (flip_horizontal && flip_vertical)
      flip_value_ = 0; // flip both, horizontal and vertical　上下水平镜像
    else if (flip_horizontal)
      flip_value_ = 1;//　水平镜像
    else if (flip_vertical)
      flip_value_ = -1;// 垂直镜像
    else
      flip_image_ = false;
// 
    camera_info_.width = image_.cols;//图像宽度
    camera_info_.height = image_.rows;//图像高度
    camera_info_.distortion_model = "plumb_bob";//数字畸变改正模型
    camera_info_.D = list_of(0)(0)(0)(0)(0).convert_to_container<std::vector<double> >();//畸变矫正　r1 r2 p1 p2 r3
    camera_info_.K = list_of(1)(0)(camera_info_.width/2)(0)(1)(camera_info_.height/2)(0)(0)(1);//相机内参数　fx 0 cx; 0 fy cy; 0 0 1
    camera_info_.R = list_of(1)(0)(0)(0)(1)(0)(0)(0)(1);//旋转矩阵 3*3
    camera_info_.P = list_of(1)(0)(camera_info_.width/2)(0)(0)(1)(camera_info_.height/2)(0)(0)(0)(1)(0);// 投影矩阵 3*4　K*[R t]
// 1s 执行一次
    timer_ = nh_.createTimer(ros::Duration(1), &ImagePublisherNodelet::do_work, this);

    dynamic_reconfigure::Server<image_publisher::ImagePublisherConfig>::CallbackType f =
      boost::bind(&ImagePublisherNodelet::reconfigureCallback, this, _1, _2);
    srv.setCallback(f);//设置　动态参数　配置　回调函数
//*/
  }
};
}
#include <pluginlib/class_list_macros.h>
//　插件
PLUGINLIB_EXPORT_CLASS(image_publisher::ImagePublisherNodelet, nodelet::Nodelet);
