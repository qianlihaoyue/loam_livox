// Author: Lin Jiarong          ziv.lin.ljr@gmail.com

// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.
// Developer: Lin Jiarong  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>

#include <pcl/point_types.h>
#define printf_line printf(" %s %d \r\n", __FILE__, __LINE__);
typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians) { return radians * 180.0 / M_PI; }

inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

#include <rclcpp/rclcpp.hpp>

rmw_qos_profile_t qos_profile{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                              1,
                              RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                              RMW_QOS_POLICY_DURABILITY_VOLATILE,
                              RMW_QOS_DEADLINE_DEFAULT,
                              RMW_QOS_LIFESPAN_DEFAULT,
                              RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                              RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                              false};

auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, qos_profile.depth), qos_profile);

rmw_qos_profile_t qos_profile_imu{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                                  2000,
                                  RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                                  RMW_QOS_POLICY_DURABILITY_VOLATILE,
                                  RMW_QOS_DEADLINE_DEFAULT,
                                  RMW_QOS_LIFESPAN_DEFAULT,
                                  RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                                  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                                  false};

auto qos_imu = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile_imu.history, qos_profile_imu.depth), qos_profile_imu);

rmw_qos_profile_t qos_profile_lidar{RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                                    5,
                                    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                                    RMW_QOS_POLICY_DURABILITY_VOLATILE,
                                    RMW_QOS_DEADLINE_DEFAULT,
                                    RMW_QOS_LIFESPAN_DEFAULT,
                                    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                                    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                                    false};

auto qos_lidar = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile_lidar.history, qos_profile_lidar.depth), qos_profile_lidar);
