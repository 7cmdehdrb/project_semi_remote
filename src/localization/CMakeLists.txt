cmake_minimum_required(VERSION 3.0.2)
project(localization)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES localization
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(pcd_to_pointcloud_node src/pcd_to_pointcloud.cpp)
add_dependencies(pcd_to_pointcloud_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pcd_to_pointcloud_node
  ${catkin_LIBRARIES}
) 

install(TARGETS pcd_to_pointcloud_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)


# HELLO WORLD!