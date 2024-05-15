cmake_minimum_required(VERSION 3.5)
project(loam_livox)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

set(dependencies
  rclcpp
  nav_msgs
  tf2_ros
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs

  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
message(STATUS "***** PCL version: ${PCL_VERSION} *****")

# ###
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
)

add_executable(livox_scanRegistration src/laser_feature_extractor.cpp)
target_link_libraries(livox_scanRegistration ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
ament_target_dependencies(livox_scanRegistration ${dependencies})

add_executable(livox_laserMapping src/laser_mapping.cpp)
target_link_libraries(livox_laserMapping ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${CERES_LIBRARIES})
ament_target_dependencies(livox_laserMapping ${dependencies})

install(
  TARGETS livox_scanRegistration
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS livox_laserMapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_package()
