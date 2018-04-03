# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( objdetect )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Haar特征 级联分类器。该分类器由若干个简单的AdaBoost强分类器串接得来
add_executable( cascade_Classifier cascade_Classifier.cpp)
target_link_libraries( cascade_Classifier ${OpenCV_LIBS} )

# LBP特征 级联分类器。该分类器由若干个简单的AdaBoost强分类器串接得来
add_executable( lbpCascade_Classifier objectDetection_lbp_feature.cpp)
target_link_libraries( lbpCascade_Classifier ${OpenCV_LIBS} )



