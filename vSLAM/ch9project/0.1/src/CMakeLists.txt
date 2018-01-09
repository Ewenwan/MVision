# 创建共享库
add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
)
# 连接 依赖库 
target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
