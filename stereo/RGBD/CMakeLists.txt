cmake_minimum_required(VERSION 2.8)

#uncomment the following for release version when need to enable PCL
option(ENABLE_PCL "Enable PCL" ON)

set(ALL_SAMPLES
    Capature_rgb_Img
    LoopDetect
    DumpAllFeatures
    SimpleView_Callback
    SimpleView_FetchFrame
    SimpleView_MultiDevice
    SimpleView_Registration
    SimpleView_TriggerMode
    SimpleView_FetchHisto
    SimpleView_TriggerMode0
    SimpleView_TriggerMode1
    SimpleView_TriggerMode_M2S1
    SimpleView_TriggerMode_M3S1
    SimpleView_KeepAlive
    )

set(SAMPLES_USED_PCL
    DepthToWorld
    SimpleView_Point3D
    )

if (NOT TARGET tycam) 
    #only build samples 
    option(ENABLE_PCL "Enable PCL" OFF)
    option (OpenCV_STATIC OFF)
    set(INCLUDE_PATH ../include)
    include_directories(${INCLUDE_PATH})
    set(TARGET_LIB tycam)
    if (MSVC)#for windows
        add_library(${TARGET_LIB} SHARED IMPORTED)
        set (LIB_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lib/win/hostapp/)
        if(CMAKE_CL_64) #x64
            set_property(TARGET ${TARGET_LIB} PROPERTY IMPORTED_LOCATION ${LIB_ROOT_PATH}/x64/tycam.dll)
            set_property(TARGET ${TARGET_LIB} PROPERTY IMPORTED_IMPLIB  ${LIB_ROOT_PATH}/x64/tycam.lib)
        else()
            set_property(TARGET ${TARGET_LIB} PROPERTY IMPORTED_LOCATION ${LIB_ROOT_PATH}/x86/tycam.dll)
            set_property(TARGET ${TARGET_LIB} PROPERTY IMPORTED_IMPLIB ${LIB_ROOT_PATH}/x86/tycam.lib)
        endif()
    endif()
endif()

if (MSVC)
    if (OpenCV_STATIC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    endif()
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# ========================================
# === common, pcl is too slow, so build a
# === lib to speed up
# ========================================
set(COMMON_SOURCES
    common/MatViewer.cpp
    common/PointCloudViewer.cpp
    )

add_library(sample_common STATIC ${COMMON_SOURCES})

# ========================================
# === OpenCV
# ========================================
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV2_INCLUDE_DIRS})
    # if(CMAKE_BUILD_TYPE STREQUAL Debug)
    # link_directories(${OpenCV_LIB_DIR_DBG})
    # else()
    # link_directories(${OpenCV_LIB_DIR_OPT})
    # endif()
    link_directories(${OpenCV_LIB_DIR})
endif()

# ========================================
# === PCL
# ========================================
# find_package(PCL 1.3 REQUIRED COMPONENTS common io)
if (ENABLE_PCL)
    find_package(PCL)
    if (NOT PCL_FOUND)
        message(WARNING "PCL library not found, some samples will not be compiled")
    else()
        add_definitions(-DHAVE_PCL)
        include_directories(${PCL_INCLUDE_DIRS})
        link_directories(${PCL_LIBRARY_DIRS})
        #message(WARNING "PCL find")
        # target_link_libraries(MY_TARGET_NAME ${PCL_LIBRARIES})
        set(ALL_SAMPLES ${ALL_SAMPLES} ${SAMPLES_USED_PCL})
    endif()
endif()


foreach(sample ${ALL_SAMPLES})
    get_filename_component(spath "${sample}" ABSOLUTE )
    if (EXISTS "${spath}/")
        file(GLOB sources ${sample}/*.cpp)
        add_executable(${sample} ${sources})
        add_dependencies(${sample} sample_common ${TARGET_LIB})
        target_link_libraries(${sample} sample_common ${TARGET_LIB} ${OpenCV_LIBS} ${PCL_LIBRARIES})
        set_target_properties(${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
    endif()
    # install(TARGETS ${sample} RUNTIME DESTINATION samples/)
endforeach()
