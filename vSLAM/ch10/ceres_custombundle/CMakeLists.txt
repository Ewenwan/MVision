cmake_minimum_required(VERSION 2.8)

project(ceres_customBundle)

find_package(Ceres REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")


include_directories(${CERES_INCLUDE_DIRS} 
                    ${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/tools 
                    ${PROJECT_SOURCE_DIR}/common/flags)

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

add_executable(${PROJECT_NAME} ceresBundle.cpp)

target_link_libraries(${PROJECT_NAME} BALProblem ParseCmd ${CERES_LIBRARIES} )
