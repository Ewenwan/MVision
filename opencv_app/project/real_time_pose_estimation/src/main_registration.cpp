/*
由简单的 长方体 顶点  面 描述的ply文件 和 物体的彩色图像 生产 物体的三维纹理模型文件
2d-3d点配准

【1】手动指定 图像中 物体顶点的位置（得到二维像素值位置）
	ply文件 中有物体定点的三维坐标

	由对应的 2d-3d点对关系
	u
	v  =  K × [R t] X
	1               Y
		        Z
		        1
	K 为图像拍摄时 相机的内参数

	世界坐标中的三维点(以文件中坐标为(0,0,0)某个定点为世界坐标系原点)
	经过 旋转矩阵R  和平移向量t 变换到相机坐标系下
	在通过相机内参数 变换到 相机的图像平面上

【2】由 PnP 算法可解的 旋转矩阵R  和平移向量t 

【3】把从图像中得到的纹理信息 加入到 物体的三维纹理模型中

	在图像中提取特征点 和对应的描述子
	利用 内参数K 、 旋转矩阵R  和平移向量t  反向投影到三维空间
	   标记 该反投影的3d点 是否在三维物体的 某个平面上

【4】将 2d-3d点对 、关键点 以及 关键点描述子 存入物体的三维纹理模型中
 
*/
// C++
#include <iostream>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
// PnP 
#include "Mesh.h"// 物体模型 读取  ply 文件   顶点坐标和 平面的顶点组成关系
#include "Model.h"// 物体网格模型  物体的 三维纹理 模型文件，包含：2d-3d点对 特征点 特征点描述子
#include "PnPProblem.h"// 2d-3d点对匹配 Rt变换求解器
#include "RobustMatcher.h"// 
#include "ModelRegistration.h"// 2d - 3d 匹配点对记录 类 
#include "Utils.h"// 画图 打印图片 显示文字等操作

using namespace cv;
using namespace std;


// 全局变量 
string tutorial_path = "../"; // path to tutorial

string img_path = tutorial_path + "Data/resized_IMG_3875.JPG";  
// 需要配准的图像  找到器三维空间位置模型

string ply_read_path = tutorial_path + "Data/box.ply";          
// 物体模型对象 object mesh  简单的 ply
//  ply文件格式详细说明  https://www.cnblogs.com/liangliangdetianxia/p/4000295.html
//  ply文件格式详细说明   https://blog.csdn.net/huapenguag/article/details/69831350

string write_path = tutorial_path + "Data/cookies_ORB.yml";     
// 输出文件 物体的 三维纹理 模型文件，包含：2d-3d点对 特征点 特征点描述子 output file



// 手动指定二位点 2d-3d 点对 配准是否完成标志  
bool end_registration = false;

// 相机内参数  K  
double f = 45; // 焦距 毫米单位 focal length in mm
double sx = 22.3, sy = 14.9; 
double width = 2592, height = 1944;//图像尺寸
double params_CANON[] = { width*f/sx,   // fx
                          height*f/sy,  // fy
                          width/2,      // cx
                          height/2};    // cy

//  ply中指定了 8个顶点 的 长方体
int n = 8;//8个顶点 
int pts[] = {1, 2, 3, 4, 5, 6, 7, 8}; // 长方体 顶点 序列

// 颜色  
Scalar red(0, 0, 255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar yellow(0,255,255);


ModelRegistration registration;//模型配准对象 记录2d-3d点对
Model model;// 物体网格模型  物体的 三维纹理 模型文件，包含：2d-3d点对 特征点 特征点描述子
Mesh mesh;  // 物体模型 读取  ply 文件   顶点坐标和 平面的顶点组成关系
PnPProblem pnp_registration(params_CANON);// 2D-3D 点对匹配 Rt变换求解器

// 帮助函声明
void help();

// 鼠标点击响应 回调函数 模型配置 
// 手动指定 图像中 物体顶点的位置（得到二维像素值位置）
static void onMouseModelRegistration( int event, int x, int y, int, void* )
{
  if  ( event == EVENT_LBUTTONUP )//鼠标按下
  {
      int n_regist = registration.getNumRegist();// 当前配准的点数 
      int n_vertex = pts[n_regist];// 顶点

      Point2f point_2d = Point2f((float)x,(float)y);// 鼠标点击的 像素位置
      Point3f point_3d = mesh.getVertex(n_vertex-1);// 对应mesh中的 三维坐标值 0~7

      // 判断当前是否还需要 确定匹配点 当前已经匹配的点对数 < 所需匹配点对数时  就还需要
      bool is_registrable = registration.is_registrable();
      if (is_registrable)
      {
        registration.registerPoint(point_2d, point_3d);//模型配准类 中添加 2d-3d配准点对
        if( registration.getNumRegist() == registration.getNumMax() ) 
		// 记录手动确定的点数 满8个就匹配完毕
		end_registration = true;
      }
  }
}


// 主函数
int main()
{
  // 帮助信息
  help();

  // 载入物体网格模型文件 *.ply  
  mesh.load(ply_read_path);

  // 设置最大 关键点个数
  int numKeyPoints = 10000;

  //鲁棒匹配器 detector, extractor, matcher
  RobustMatcher rmatcher;
  Ptr<FeatureDetector> detector = ORB::create(numKeyPoints);//ORB特征点检测器
  rmatcher.setFeatureDetector(detector);//匹配器的特征 提取方法

  // 创建显示窗口
  // Create & Open Window
  namedWindow("MODEL REGISTRATION", WINDOW_KEEPRATIO);
  // 鼠标点击事件 Set up the mouse events
  setMouseCallback("MODEL REGISTRATION", onMouseModelRegistration, 0 );

  // 读取图像进行匹配 Open the image to register
  Mat img_in = imread(img_path, IMREAD_COLOR);
  Mat img_vis = img_in.clone();

  if (!img_in.data) {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  // 设置需要手动 配准的 2d-3d点对数
  int num_registrations = n;//初始最大需要配置的 点对数
  registration.setNumMax(num_registrations);

  cout << "Click the box corners ..." << endl;
  cout << "Waiting ..." << endl;

//======循环直到  完成配准 Loop until all the points are registered
  while ( waitKey(30) < 0 )
  {
    // Refresh debug image
    img_vis = img_in.clone();//更新图像

    // 得到当前配准的点对 2d-3d Current registered points
    vector<Point2f> list_points2d = registration.get_points2d();
    vector<Point3f> list_points3d = registration.get_points3d();

    // 显示当前配准的点对 2d-3d Draw current registered points
    drawPoints(img_vis, list_points2d, list_points3d, red);// 红色 2d点

    if (!end_registration)//配准未完成 显示还需要配准的3d点
    {
      // Draw debug text
      int n_regist = registration.getNumRegist();// 已经匹配的点
      int n_vertex = pts[n_regist];//对应三维点的 索引
      Point3f current_poin3d = mesh.getVertex(n_vertex-1);//对应ply文件中 三维点的坐标 

      drawQuestion(img_vis, current_poin3d, green);// 绿色 3d反投影的点
      drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), red);
    }
    else//已经配准完成 结束循环
    {
      // Draw debug text
      drawText(img_vis, "END REGISTRATION", green);
      drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), green);
      break;//  结束循环
    }

    // Show the image
    imshow("MODEL REGISTRATION", img_vis);
  }

//=====配准完成 进行相机位姿 计算==============

  cout << "COMPUTING POSE ..." << endl;

  // 匹配的 2d-3d点对   
  vector<Point2f> list_points2d = registration.get_points2d();
  vector<Point3f> list_points3d = registration.get_points3d();

//=====pnp 估计变换矩阵  ========== 
  bool is_correspondence = pnp_registration.estimatePose(list_points3d, list_points2d, SOLVEPNP_ITERATIVE);
  if ( is_correspondence )
  {
    cout << "Correspondence found" << endl;

//=====使用得到的 变换位姿 将 三维网格中的3d点 投影到 2d点 并显示===========
    vector<Point2f> list_points2d_mesh = pnp_registration.verify_points(&mesh);
    draw2DPoints(img_vis, list_points2d_mesh, green);//显示3d -> 2d点 绿色

  } else {
    cout << "Correspondence not found" << endl << endl;
  }
  //  显示图像
  imshow("MODEL REGISTRATION", img_vis);
  //  ESC 键 结束
  waitKey(0);


//===计算图像2d特征点，使用变换矩阵得到3d点  ===================
  vector<KeyPoint> keypoints_model;//关键点
  Mat descriptors;//描述子
  rmatcher.computeKeyPoints(img_in, keypoints_model);// 检测关键点
  rmatcher.computeDescriptors(img_in, keypoints_model, descriptors);//计算关键点的描述子

  for (unsigned int i = 0; i < keypoints_model.size(); ++i) {
    Point2f point2d(keypoints_model[i].pt);//每一个检测出的 2d 特征点
    Point3f point3d;// 按变换关系 转换成的三维点
    bool on_surface = pnp_registration.backproject2DPoint(&mesh, point2d, point3d);
    if (on_surface)// 转换得到的3d点 在物体表面上
    {
        model.add_correspondence(point2d, point3d);// 模型文件添加2d-3d点
        model.add_descriptor(descriptors.row(i));//描述子
        model.add_keypoint(keypoints_model[i]);//关键点
    }
    else// 转换得到的3d点  不在物体表面上
    {
        model.add_outlier(point2d);// 模型文件添加2d外点
    }
  }

  // 保存模型文件 *.yaml 
  model.save(write_path);

  // 输出图像
  img_vis = img_in.clone();

  // 2d - 3d 点 
  vector<Point2f> list_points_in = model.get_points2d_in();
  vector<Point2f> list_points_out = model.get_points2d_out();

  // 显示一些文字  内点 和 外点数量
  string num = IntToString((int)list_points_in.size());
  string text = "There are " + num + " inliers";//内点
  drawText(img_vis, text, green);

  num = IntToString((int)list_points_out.size());
  text = "There are " + num + " outliers";//外点
  drawText2(img_vis, text, red);

  // 画物体三维网格 绿色
  drawObjectMesh(img_vis, &mesh, &pnp_registration, blue);

  //显示找到的 关键点  内点显示绿色  外点显示红色
  draw2DPoints(img_vis, list_points_in, green);
  draw2DPoints(img_vis, list_points_out, red);

  // 显示最后的图像
  imshow("MODEL REGISTRATION", img_vis);

  //  ESC 键 结束
  waitKey(0);

  // 是否窗口
  destroyWindow("MODEL REGISTRATION");

  cout << "GOODBYE" << endl;

}

/**********************************************************************************************************/
void help()
{
  cout
  << "--------------------------------------------------------------------------"   << endl
  << "This program shows how to create your 3D textured model. "                    << endl
  << "Usage:"                                                                       << endl
  << "./cpp-tutorial-pnp_registration"                                              << endl
  << "--------------------------------------------------------------------------"   << endl
  << endl;
}
