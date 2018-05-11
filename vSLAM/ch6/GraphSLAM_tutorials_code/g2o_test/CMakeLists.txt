cmake_minimum_required(VERSION 2.8)
project(g2o_test)

#Enable support for C++11
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

#设定二进制文件路径
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#设定库文件编译路径
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#设定.cmake文件路径
#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "{CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
#ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src)

# find g2o lib
find_package(G2O REQUIRED)
IF(G2O_FOUND)
    include_directories(${G2O_INCLUDE_DIR})
    message("G2O lib is found:"${G2O_INCLUDE_DIR})
ENDIF(G2O_FOUND)

find_package(Eigen3 REQUIRED)
find_package(CSparse REQUIRED)
#find_package(Cholmod REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(${CHOLMOD_INCLUDE_DIR})

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

ADD_EXECUTABLE(g2o_test main.cpp)
target_link_libraries(g2o_test ${G2O_LIBS})
#target_link_libraries(g2o_test csparse g2o_core g2o_solver_cholmod g2o_solver_csparse g2o_types_slam3d)
