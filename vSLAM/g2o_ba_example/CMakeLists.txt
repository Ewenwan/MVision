# CMake 工程，读者应该熟悉了，我就不一一注释了
cmake_minimum_required( VERSION 2.8 )
project( g2o_ba_example )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3 -march=native" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#SET(CMAKE_BUILD_TYPE "Debug") 
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# cmake  -DCMAKE_BUILD_TYPE=Debug ..   make
# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
# 找 opencv'库 3.1
find_package( OpenCV 3.1 REQUIRED )
#find_package( Eigen3 REQUIRED )
#find_package( Cholmod )
find_package( CSparse REQUIRED )
# 包含库
include_directories( 
    ${OpenCV_INCLUDE_DIRS} #opencv
    ${G2O_INCLUDE_DIRS} #G2O
    ${CSPARSE_INCLUDE_DIR} #CSparse  稀疏矩阵求解
#    ${CHOLMOD_INCLUDE_DIR}
    "/usr/include/eigen3/" #Eigen3
)

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

add_executable( ba_example g2o_ba_example.cpp )
target_link_libraries( ba_example 
    ${OpenCV_LIBS}
    #${G2O_LIBS}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension cxsparse
    ${CHOLMOD_LIBRARIES}
    ${CSPARSE_LIBRARIES}

    )
