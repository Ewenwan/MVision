# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( machine_learning )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 支持向量机
# L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)
add_executable( svm support_Vector_Machines.cpp)
target_link_libraries( svm ${OpenCV_LIBS} )

# 支持向量机对线性不可分数据的处理
# 考虑 错分类样本离同类区域的距离
add_executable( nLinearsvm nLinearlySvm.cpp)
target_link_libraries(  nLinearsvm ${OpenCV_LIBS} )

# 主成分分析  数据降维
add_executable( principal_C_A  principal_Component_Analysis.cpp)
target_link_libraries( principal_C_A  ${OpenCV_LIBS} )



