#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_recognition )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT 
common io visualization 
features  keypoints filters
recognition kdtree 
)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


# 基于对应分组的三维物体识别
#【1】计算法线向量
#【2】下采样滤波使用均匀采样（可以试试体素格子下采样）得到关键点
#【3】为keypoints关键点计算SHOT描述子
#【4】按存储方法KDTree匹配两个点云（描述子向量匹配）点云分组得到匹配的组 描述 点对匹配关系
#【5】参考帧霍夫聚类/集合一致性聚类得到 匹配点云cluster  平移矩阵和 匹配点对关系
add_executable( correspondence_grouping correspondence_grouping.cpp)
target_link_libraries( correspondence_grouping  ${PCL_LIBRARIES} )

# 隐式形状模型 ISM （隐形状模型 （Implicit Shape Model））
# 训练模型　识别模型点云
add_executable(implicit_shape_model implicit_shape_model.cpp)
target_link_libraries( implicit_shape_model  ${PCL_LIBRARIES} )

# 3D物体识别的假设检验 对应分组的三维物体识别 icp点云配准　验证结果
add_executable( hVObjectRec hypothesis_Ver_Object_Rec.cpp)
target_link_libraries( hVObjectRec  ${PCL_LIBRARIES} )

