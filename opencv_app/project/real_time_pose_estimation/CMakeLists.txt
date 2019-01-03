# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( processingImage )
# 找opencv
find_package( OpenCV REQUIRED  )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/)
set(target cpp-tutorial-)

set(sample_pnplib
        ${sample_dir}CsvReader.cpp
        ${sample_dir}CsvWriter.cpp
        ${sample_dir}ModelRegistration.cpp
        ${sample_dir}Mesh.cpp
        ${sample_dir}Model.cpp
        ${sample_dir}PnPProblem.cpp
        ${sample_dir}Utils.cpp
        ${sample_dir}RobustMatcher.cpp
)

# 生成物体三维纹理模型数据
add_executable( ${target}pnp_registration ${sample_dir}main_registration.cpp ${sample_pnplib} )

# 使用数据库 对视频/拍摄图像 进行实时 检测 目标定位
add_executable( ${target}pnp_detection ${sample_dir}main_detection.cpp ${sample_pnplib} )

# target_link_libraries( ${target}pnp_registration ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} )
target_link_libraries( ${target}pnp_registration ${OpenCV_LIBS} )
target_link_libraries( ${target}pnp_detection    ${OpenCV_LIBS})
