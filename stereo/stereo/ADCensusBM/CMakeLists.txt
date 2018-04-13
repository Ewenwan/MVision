# cmake　版本要求
cmake_minimum_required(VERSION 2.8)
#　工程名
project(ADCensusBM)

# 其他cmake模块 
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_module")
# 找并行计算框架库　OpenMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# opencv
find_package( OpenCV 3 REQUIRED  )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )
SET(LIBS ${OpenCV_LIBS})
if(OPENCV_FOUND)
    message("OPENCV FOUND")
endif()

#  boost
if(WIN32)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC ON)
  message("WIN32 ")
endif(WIN32)
SET(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.40.0 COMPONENTS filesystem system REQUIRED)
set(BOOST_FOUND ${Boost_FOUND})
set(BOOST_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
set(BOOST_LIBRARY_DIRS "${Boost_LIBRARY_DIRS}")
set(BOOST_LIBRARIES ${Boost_LIBRARIES})
INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_DATE_TIME_LIBRARY})

# Under Windows the system variable "PCL_ROOT" must be set to the location of the root directory of PCL.
# pcl点云库
find_package(PCL 1.2 REQUIRED)
SET(LIBS ${LIBS} ${PCL_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#  libconfig++ 配置文件读取库
FIND_PACKAGE(Config++ REQUIRED)
INCLUDE_DIRECTORIES(${LIBCONFIG_INCLUDE_DIR})
SET(LIBS ${LIBS} ${LIBCONFIG_LIBRARIES})

#aux_source_directory(. SRC_LIST)# 查找在某个路径下的所有源文件
set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/)
SET(SRC_LIST 
${sample_dir}imageprocessor.cpp
${sample_dir}stereoprocessor.cpp
${sample_dir}adcensuscv.cpp
${sample_dir}scanlineoptimization.cpp
${sample_dir}aggregation.cpp
${sample_dir}disparityrefinement.cpp
)

add_executable(adcense stereo_match_bm_sgbm_adcense.cpp ${SRC_LIST})
#set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "USE_PCL=1")
target_link_libraries(adcense ${LIBS} ${OpenCV_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})




