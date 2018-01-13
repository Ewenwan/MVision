/*
 * 局部 3D -2D 点匹配算法     其中的3D点是当前帧图像 相机坐标系下的 点
 * 全局 3D -2D 点匹配算法     其中的3D点是第一帧帧图像 相机坐标系(世界坐标系)下的 点
 * 匹配算法 也是当前描述子和  地图描述子匹配 对应的 3D点
 * 地图点 第一帧 的3D点全部加入地图 此后 新的一帧 图像在地图中 没有匹配到的 像素点 转换成世界坐标系下的三维点后 加入地图
 * 同时对地图优化 不在当前帧的点（看不见的点） 删除 匹配次数不高的 点   删除 视角过大删除
 */
/*在求解相机位姿之后使用非线性图优化*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{
    //【1】默认默认构造函数
      VisualOdometry::VisualOdometry() :
	  state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_( new cv::flann::LshIndexParams(5,10,2) )
      {
	num_of_features_    = Config::get<int> ( "number_of_features" );// 特征数量   整型 int
	scale_factor_       = Config::get<double> ( "scale_factor" );            // 尺度因子 缩小
	level_pyramid_      = Config::get<int> ( "level_pyramid" );              // 层级 
	match_ratio_        = Config::get<float> ( "match_ratio" );               // 匹配 参数
	max_num_lost_       = Config::get<float> ( "max_num_lost" );       // 最大丢失次数
	min_inliers_        = Config::get<int> ( "min_inliers" );       		// 最小内点数量
	key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );//最小的旋转      成为关键帧的条件
	key_frame_min_trans = Config::get<double> ( "keyframe_translation" ); // 最小平移  
	map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");
	orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );//创建orb特征
	
      }
    // 【2】析构函数   shared_ptr 智能指针 自带析构函数功能
      VisualOdometry::~VisualOdometry()
      {  }

      bool VisualOdometry::addFrame ( Frame::Ptr frame )
      {
	  switch ( state_ )
	  {
	  case INITIALIZING:
	  {
	     state_ = OK;//初始化之后切换状态为 OK
	     curr_ = ref_ = frame;//初始化 参考帧 当前帧
	     map_->insertKeyFrame ( frame );//添加第一个关键帧 到地图内
	    // extract features from first frame 
	    extractKeyPoints();//提取关键点
	    computeDescriptors();//计算描述子
	    setRef3DPoints();//计算特征点的 3维坐标
	      break;
	  }
	  case OK:
	  {
              curr_ = frame;//新的一帧为当前帧
	      extractKeyPoints();//提取关键点
	      computeDescriptors();//计算描述子
	      featureMatching();      //特征点匹配 得到 特征点匹配点对
	      poseEstimationPnP();//位置估计 加入非线性优化算法 根据特征点匹配点对  计算坐标转换 及估计相机位姿
	      if ( checkEstimatedPose() == true ) // a good estimation
	      {
		  curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r * T_r_w 
                  ref_ = curr_;//迭代当前帧成参考帧
		  setRef3DPoints();//计算特征点的 3维坐标  
		  num_lost_ = 0;//重置 丢失计数
		  if ( checkKeyFrame() == true ) // is a key-frame
		  {
		      addKeyFrame();
		  }
	      }
	      else // bad estimation due to various reasons
	      {
		  num_lost_++;
		  if ( num_lost_ > max_num_lost_ )
		  {
		      state_ = LOST;
		  }
		  return false;
	      }
	      break;
	  }
	  case LOST:
	  {
	      cout<<"vo has lost."<<endl;
	      break;
	  }
	  }

	  return true;
      }
      
      
      // 提取关键点
      void VisualOdometry::extractKeyPoints()
      {
	  boost::timer timer;
	  orb_->detect ( curr_->color_, keypoints_curr_ );
	  cout<<"extract keypoints cost time: "<<timer.elapsed()<<endl;
      }
      //计算描述子
      void VisualOdometry::computeDescriptors()
      {
	  boost::timer timer;
	  orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
	  cout<<"descriptor computation cost time: "<<timer.elapsed()<<endl;
      }
      //特征点描述子匹配
      void VisualOdometry::featureMatching()
      {
	  boost::timer timer;
	  vector<cv::DMatch> matches;
	  matcher_flann_.match( descriptors_ref_, descriptors_curr_, matches );
	  // select the best matches
	  float min_dis = std::min_element (
			      matches.begin(), matches.end(),
			      [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
	  {
	      return m1.distance < m2.distance;
	  } )->distance;

	  feature_matches_.clear();
	  for ( cv::DMatch& m : matches )
	  {
	      if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
	      {
		  feature_matches_.push_back(m);
	      }
	  }
	  cout<<"good matches: "<<feature_matches_.size()<<endl;
	  cout<<"match cost time: "<<timer.elapsed()<<endl;
      }
      
      //计算特征点 第二帧相机坐标系下的 三维坐标 为 局部匹配算法  计算得到相对上一帧图像 的 平移矩阵
      void VisualOdometry::setRef3DPoints()
      {
	  // select the features with depth measurements 
	  pts_3d_ref_.clear();
	  descriptors_ref_ = Mat();
	  for ( size_t i=0; i<keypoints_curr_.size(); i++ )
	  {
	      double d = ref_->findDepth(keypoints_curr_[i]);               
	      if ( d > 0)
	      {
		  Vector3d p_cam = ref_->camera_->pixel2camera(
		      Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d//第二幅图像相机坐标系 下的三维坐标
		  );
		  pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
		  descriptors_ref_.push_back(descriptors_curr_.row(i));
	      }
	  }
      }

      //位置估计 加入非线性优化算法
      void VisualOdometry::poseEstimationPnP()
      {
	  // construct the 3d 2d observations
	  vector<cv::Point3f> pts3d;
	  vector<cv::Point2f> pts2d;
	  
	  for ( cv::DMatch m:feature_matches_ )
	  {
	      pts3d.push_back( pts_3d_ref_[m.queryIdx] );
	      pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
	  }
	  
	  Mat K = ( cv::Mat_<double>(3,3)<<
	      ref_->camera_->fx_, 0, ref_->camera_->cx_,
	      0, ref_->camera_->fy_, ref_->camera_->cy_,
	      0,0,1
	  );
	  Mat rvec, tvec, inliers;
	  // PnP算法求解  2D-3D点对求解 旋转向量 rvec,  平移矩阵 tvec
	  cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );// 随机采样序列 算法  得到符合求解得到的R t 的点的数量
	  num_inliers_ = inliers.rows;
	  cout<<"pnp inliers: "<<num_inliers_<<endl;//内点数量
	  T_c_r_estimated_ = SE3(
	      SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
	      Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
	  );
	  
	  //在上述PNP求解之后加入图优化
	  // using bundle adjustment to optimize the pose 
	  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;//矩阵块 分解 求解器
	  Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
	  Block* solver_ptr = new Block( linearSolver );
	  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );//LM迭代算法
	  g2o::SparseOptimizer optimizer;
	  optimizer.setAlgorithm ( solver );//非线性优化器
	  
	  // 添加顶点
	  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();//顶点  位姿
	  pose->setId ( 0 );
	  pose->setEstimate ( g2o::SE3Quat (
	      T_c_r_estimated_.rotation_matrix(), //旋转矩阵
	      T_c_r_estimated_.translation()//平移向量
	  ) );
	  optimizer.addVertex ( pose );//加入优化列队

	  // 添加 边 edges  在 随机采样序列 算法中 符合 求得的 RT的点对
	  for ( int i=0; i<inliers.rows; i++ )
	  {
	      int index = inliers.at<int>(i,0);
	      // 3D -> 2D projection
	      EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();// 仅优化 pose
	      edge->setId(i);
	      edge->setVertex(0, pose);//顶点
	      edge->camera_ = curr_->camera_.get();
	      edge->point_ = Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z );
	      edge->setMeasurement( Vector2d(pts2d[index].x, pts2d[index].y) );//测量值为 2D点 另一帧图像的 特征点 像素坐标
	      edge->setInformation( Eigen::Matrix2d::Identity() );
	      optimizer.addEdge( edge );//添加边
	  }
	  
	  optimizer.initializeOptimization();//初始化优化器
	  optimizer.optimize(10);//执行10次优化
	  
	  T_c_r_estimated_ = SE3 (
	      pose->estimate().rotation(),
	      pose->estimate().translation()
	  );
      }
      //检查估计的位姿
      bool VisualOdometry::checkEstimatedPose()
      {
	  // check if the estimated pose is good
	  if ( num_inliers_ < min_inliers_ )
	  {
	      cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
	      return false;
	  }
	  // if the motion is too large, it is probably wrong
	  Sophus::Vector6d d = T_c_r_estimated_.log();
	  if ( d.norm() > 5.0 )
	  {
	      cout<<"reject because motion is too large: "<<d.norm()<<endl;
	      return false;
	  }
	  return true;
      }

      //检查是否为关键帧
      bool VisualOdometry::checkKeyFrame()
      {
	  Sophus::Vector6d d = T_c_r_estimated_.log();
	  Vector3d trans = d.head<3>();
	  Vector3d rot = d.tail<3>();
	  if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
	      return true;
	  return false;
      }

      void VisualOdometry::addKeyFrame()
      {
	  cout<<"adding a key-frame"<<endl;
	  map_->insertKeyFrame ( curr_ );
      }

}
