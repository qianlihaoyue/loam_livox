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

#ifndef LASER_FEATURE_EXTRACTION_H
#define LASER_FEATURE_EXTRACTION_H

#include <cmath>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "rclcpp/rclcpp.hpp"
#include <rclcpp/subscription.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <vector>

#include "livox_feature_extractor.hpp"
#include "tools/common.h"
#include "tools/logger.hpp"

using std::atan2;
using std::cos;
using std::sin;
using namespace Common_tools;

class Laser_feature : public rclcpp::Node {
public:
    const double m_para_scanPeriod = 0.1;

    int m_if_pub_debug_feature = 1;

    const int m_para_system_delay = 2;
    int m_para_system_init_count = 0;
    bool m_para_systemInited = false;
    float m_pc_curvature[400000];
    int m_pc_sort_idx[400000];
    int m_pc_neighbor_picked[400000];
    int m_pc_cloud_label[400000];
    int m_if_motion_deblur = 0;
    int m_odom_mode = 0;  // 0 = for odom, 1 = for mapping
    float m_plane_resolution;
    float m_line_resolution;
    File_logger m_file_logger;

    bool m_if_pub_each_line = false;
    int m_lidar_type = 0;  // 0 is velodyne, 1 is livox
    int m_laser_scan_number = 64;
    Livox_laser m_livox;
    rclcpp::Time m_init_timestamp;
    bool comp(int i, int j) { return (m_pc_curvature[i] < m_pc_curvature[j]); }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_laser_pc;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_sharp_corner;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_less_sharp_corner;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_surface_flat;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_surface_less_flat;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_removed_pt;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> m_pub_each_scan;

    string lid_topic;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_sub_input_laser_cloud;

    double MINIMUM_RANGE = 0.01;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pub_pc_livox_corners, m_pub_pc_livox_surface, m_pub_pc_livox_full;
    sensor_msgs::msg::PointCloud2 temp_out_msg;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_surface;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_corner;

    Laser_feature(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("scanRegistration", options) {
        m_init_timestamp = this->get_clock()->now();
    
        this->declare_parameter<std::string>("lid_topic", "laser_points");
        this->declare_parameter<std::string>("lidar_type", "livox");
        this->declare_parameter<int>("scan_line", 16);
        this->declare_parameter<float>("mapping_plane_resolution", 0.8);
        this->declare_parameter<float>("mapping_line_resolution", 0.8);
        this->declare_parameter<double>("minimum_range", 0.1);
        this->declare_parameter<int>("if_motion_deblur", 1);
        this->declare_parameter<int>("odom_mode", 0);
        this->declare_parameter<double>("corner_curvature", 0.05);
        this->declare_parameter<double>("surface_curvature", 0.01);
        this->declare_parameter<double>("minimum_view_angle", 10);

        this->get_parameter("lid_topic", lid_topic);
        this->get_parameter("scan_line", m_laser_scan_number);
        this->get_parameter("mapping_plane_resolution", m_plane_resolution);
        this->get_parameter("mapping_line_resolution", m_line_resolution);
        this->get_parameter("minimum_range", MINIMUM_RANGE);
        this->get_parameter("if_motion_deblur", m_if_motion_deblur);
        this->get_parameter("odom_mode", m_odom_mode);

        double livox_corners, livox_surface, minimum_view_angle;
        this->get_parameter("corner_curvature", livox_corners);
        this->get_parameter("surface_curvature", livox_surface);
        this->get_parameter("minimum_view_angle", minimum_view_angle);

        init_livox_lidar_para();

        m_livox.thr_corner_curvature = livox_corners;
        m_livox.thr_surface_curvature = livox_surface;
        m_livox.minimum_view_angle = minimum_view_angle;

        printf("scan line number %d \n", m_laser_scan_number);

        if (m_laser_scan_number != 16 && m_laser_scan_number != 64) {
            printf("only support velodyne with 16 or 64 scan line!");
            return;
        }

        std::string log_save_dir_name;
        this->declare_parameter<std::string>("log_save_dir", "../");
        this->get_parameter("log_save_dir", log_save_dir_name);

        m_file_logger.set_log_dir(log_save_dir_name);
        m_file_logger.init("scanRegistration.log");

        auto cloud_callback = [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) -> void { laserCloudHandler(msg); };
        m_sub_input_laser_cloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SystemDefaultsQoS(), cloud_callback);

        m_pub_laser_pc = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_points_2", 1);
        m_pub_pc_sharp_corner = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_sharp", 1);
        m_pub_pc_less_sharp_corner = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_less_sharp", 1);
        m_pub_pc_surface_flat = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_flat", 1);
        m_pub_pc_surface_less_flat = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_less_flat", 1);
        m_pub_pc_removed_pt = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_remove_points", 1);

        m_pub_pc_livox_corners = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pc2_corners", 1);
        m_pub_pc_livox_surface = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pc2_surface", 1);
        m_pub_pc_livox_full = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pc2_full", 1);

        m_voxel_filter_for_surface.setLeafSize(m_plane_resolution / 2, m_plane_resolution / 2, m_plane_resolution / 2);
        m_voxel_filter_for_corner.setLeafSize(m_line_resolution, m_line_resolution, m_line_resolution);

        if (m_if_pub_each_line) {
            for (int i = 0; i < m_laser_scan_number; i++) {
                auto tmp = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
                m_pub_each_scan.push_back(tmp);
            }
        }
    }

    ~Laser_feature(){};

    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float thres) {
        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i) {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z <
                thres * thres)
                continue;

            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }

        if (j != cloud_in.points.size()) {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &laserCloudMsg) {
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(m_laser_scan_number);

        if (!m_para_systemInited) {
            m_para_system_init_count++;

            if (m_para_system_init_count >= m_para_system_delay) {
                m_para_systemInited = true;
            } else
                return;
        }

        std::vector<int> scanStartInd(1000, 0);
        std::vector<int> scanEndInd(1000, 0);

        pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
        pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
        int raw_pts_num = laserCloudIn.size();

        m_file_logger.printf(" Time: %.5f, num_raw: %d, num_filted: %d\r\n", laserCloudMsg->header.stamp.nanosec * 1e-9, raw_pts_num, laserCloudIn.size());

        size_t cloudSize = laserCloudIn.points.size();

        if (m_lidar_type)  // Livox scans
        {
            laserCloudScans = m_livox.extract_laser_features(laserCloudIn, laserCloudMsg->header.stamp.nanosec * 1e-9);

            if (laserCloudScans.size() <= 5)  // less than 5 scan
            {

                return;
            }

            m_laser_scan_number = laserCloudScans.size() * 1.0;

            scanStartInd.resize(m_laser_scan_number);
            scanEndInd.resize(m_laser_scan_number);
            std::fill(scanStartInd.begin(), scanStartInd.end(), 0);
            std::fill(scanEndInd.begin(), scanEndInd.end(), 0);

            if (m_if_pub_debug_feature) {
                /********************************************
                 *    Feature extraction for livox lidar     *
                 ********************************************/
                int piece_wise = 3;

                vector<float> piece_wise_start(piece_wise);
                vector<float> piece_wise_end(piece_wise);

                for (int i = 0; i < piece_wise; i++) {
                    int start_scans, end_scans;

                    start_scans = int((m_laser_scan_number * (i)) / piece_wise);
                    end_scans = int((m_laser_scan_number * (i + 1)) / piece_wise) - 1;

                    int end_idx = laserCloudScans[end_scans].size() - 1;
                    piece_wise_start[i] = ((float)m_livox.find_pt_info(laserCloudScans[start_scans].points[0])->idx) / m_livox.m_pts_info_vec.size();
                    piece_wise_end[i] = ((float)m_livox.find_pt_info(laserCloudScans[end_scans].points[end_idx])->idx) / m_livox.m_pts_info_vec.size();
                }

                for (int i = 0; i < piece_wise; i++) {
                    pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>()), livox_surface(new pcl::PointCloud<PointType>()),
                        livox_full(new pcl::PointCloud<PointType>());

                    m_livox.get_features(*livox_corners, *livox_surface, *livox_full, piece_wise_start[i], piece_wise_end[i]);

                    rclcpp::Time current_time = this->get_clock()->now();

                    pcl::toROSMsg(*livox_full, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    m_pub_pc_livox_full->publish(temp_out_msg);

                    m_voxel_filter_for_surface.setInputCloud(livox_surface);
                    m_voxel_filter_for_surface.filter(*livox_surface);
                    pcl::toROSMsg(*livox_surface, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    m_pub_pc_livox_surface->publish(temp_out_msg);

                    m_voxel_filter_for_corner.setInputCloud(livox_corners);
                    m_voxel_filter_for_corner.filter(*livox_corners);
                    pcl::toROSMsg(*livox_corners, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    m_pub_pc_livox_corners->publish(temp_out_msg);
                    if (m_odom_mode == 0)  // odometry mode
                    {
                        break;
                    }
                }
            }
            return;
        } else {
            /********************************************
             *    Feature extraction for velodyne lidar *
             ********************************************/
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
            removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

            // printf_line;
            float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
            float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

            if (endOri - startOri > 3 * M_PI) {
                endOri -= 2 * M_PI;
            } else if (endOri - startOri < M_PI) {
                endOri += 2 * M_PI;
            }

            // printf("end Ori %f\n", endOri);

            // printf_line;
            bool halfPassed = false;
            int count = cloudSize;
            PointType point;

            for (size_t i = 0; i < cloudSize; i++) {
                point.x = laserCloudIn.points[i].x;
                point.y = laserCloudIn.points[i].y;
                point.z = laserCloudIn.points[i].z;

                float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
                int scanID = 0;

                if (m_laser_scan_number == 16) {
                    scanID = int((angle + 15) / 2 + 0.5);

                    if (scanID > (m_laser_scan_number - 1) || scanID < 0) {
                        count--;
                        continue;
                    }
                } else if (m_laser_scan_number == 64) {
                    if (angle >= -8.83) {
                        scanID = int((2 - angle) * 3.0 + 0.5);
                    } else {
                        scanID = m_laser_scan_number / 2 + int((-8.83 - angle) * 2.0 + 0.5);
                    }

                    // use [0 50]  > 50 remove outlies
                    if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                        count--;
                        continue;
                    }
                } else {
                    printf("wrong scan number\n");
                    // ROS_BREAK();
                    assert(false);
                }

                // printf("angle %f scanID %d \n", angle, scanID);
                // printf_line;
                float ori = -atan2(point.y, point.x);

                if (!halfPassed) {
                    if (ori < startOri - M_PI / 2) {
                        ori += 2 * M_PI;
                    } else if (ori > startOri + M_PI * 3 / 2) {
                        ori -= 2 * M_PI;
                    }

                    if (ori - startOri > M_PI) {
                        halfPassed = true;
                    }
                } else {
                    ori += 2 * M_PI;

                    if (ori < endOri - M_PI * 3 / 2) {
                        ori += 2 * M_PI;
                    } else if (ori > endOri + M_PI / 2) {
                        ori -= 2 * M_PI;
                    }
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = scanID + m_para_scanPeriod * relTime;
                laserCloudScans[scanID].push_back(point);
            }

            // printf_line;
            // printf( "points size %d \n", cloudSize );
        }

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        laserCloud->clear();
        cloudSize = 0;
        for (int i = 0; i < m_laser_scan_number; i++) {
            scanStartInd[i] = laserCloud->size() + 5;
            //*laserCloud += laserCloudScans[ laserCloudScans.size() - N_SCANS + i ];
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
            // cloudSize += laserCloudScans[ laserCloudScans.size() - N_SCANS + i ].size();
            cloudSize += laserCloudScans[i].size();
        }
        // printf_line;

        for (size_t i = 5; i < cloudSize - 5; i++) {
            // assert( laserCloud->points[ i - 5 ].x != 0 );
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x +
                          laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x +
                          laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y +
                          laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y +
                          laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z +
                          laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z +
                          laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
            m_pc_curvature[i] = diff;
            m_pc_sort_idx[i] = i;
            m_pc_neighbor_picked[i] = 0;
            m_pc_cloud_label[i] = 0;
            if (1) {
                if (1) {
                    if (diff > 0.1) {
                        float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + laserCloud->points[i].y * laserCloud->points[i].y +
                                            laserCloud->points[i].z * laserCloud->points[i].z);
                        float depth2 =
                            sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                                 laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

                        if (depth1 > depth2) {
                            diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
                            diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
                            diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

                            if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
                                m_pc_neighbor_picked[i - 5] = 1;
                                m_pc_neighbor_picked[i - 4] = 1;
                                m_pc_neighbor_picked[i - 3] = 1;
                                m_pc_neighbor_picked[i - 2] = 1;
                                m_pc_neighbor_picked[i - 1] = 1;
                                m_pc_neighbor_picked[i] = 1;
                            }
                        } else {
                            diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
                            diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
                            diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

                            if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
                                m_pc_neighbor_picked[i + 1] = 1;
                                m_pc_neighbor_picked[i + 2] = 1;
                                m_pc_neighbor_picked[i + 3] = 1;
                                m_pc_neighbor_picked[i + 4] = 1;
                                m_pc_neighbor_picked[i + 5] = 1;
                                m_pc_neighbor_picked[i + 6] = 1;
                            }
                        }
                    }
                }

                if (1) {
                    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
                    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
                    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
                    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

                    float dis = laserCloud->points[i].x * laserCloud->points[i].x + laserCloud->points[i].y * laserCloud->points[i].y +
                                laserCloud->points[i].z * laserCloud->points[i].z;

                    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
                        m_pc_neighbor_picked[i] = 1;
                    }
                }
            }
        }
        // printf_line;

#if !IF_LIVOX_HANDLER_REMOVE
        if (m_lidar_type != 0)
        // if(1)
        {
            Livox_laser::Pt_infos *pt_info;
            for (unsigned int idx = 0; idx < cloudSize; idx++) {
                // printf( "Idx = %d, size = %d, pt = [%f,%f,%f]\r\n", idx, cloudSize, laserCloud->points[ idx ].x, laserCloud->points[ idx ].y,
                // laserCloud->points[ idx ].z );
                //  printf( "Idx = %d, pt = [%f,%f,%f]\r\n", idx, 1.0,2.0,3.0 );
                pt_info = m_livox.find_pt_info(laserCloud->points[idx]);

                if (pt_info->pt_type != Livox_laser::e_pt_normal) {
                    // std::cout << "Reject, id = "<<idx << " ---, type = " << livox.m_mask_pointtype[idx] <<std::endl;
                    m_pc_neighbor_picked[idx] = 1;
                }
            }
        }
// printf_line;
#endif

        // printf_line;

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;
        float sharp_point_threshold = 0.05;

        // extract corners points and surface points
        for (int i = 0; i < m_laser_scan_number; i++) {
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            // To ensure the distribution of features point, spilt each scan into 6 parts equally according to their curvature.
            for (int j = 0; j < 6; j++) {
                // Starting of each sub-scan.
                int sp = (scanStartInd[i] * (6 - j) + scanEndInd[i] * j) / 6;
                // Ending of each sub-scan.
                int ep = (scanStartInd[i] * (5 - j) + scanEndInd[i] * (j + 1)) / 6 - 1;

                // sort curvature
                for (int k = sp + 1; k <= ep; k++) {
                    for (int l = k; l >= sp + 1; l--) {
                        if (m_pc_curvature[m_pc_sort_idx[l]] < m_pc_curvature[m_pc_sort_idx[l - 1]]) {
                            int temp = m_pc_sort_idx[l - 1];
                            m_pc_sort_idx[l - 1] = m_pc_sort_idx[l];
                            m_pc_sort_idx[l] = temp;
                        }
                    }
                }

                // select the most shart and flat point
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = m_pc_sort_idx[k];  // The index of biggest curvature.

                    if (m_pc_neighbor_picked[ind] == 0 && m_pc_curvature[ind] > sharp_point_threshold * 10) {
                        largestPickedNum++;
                        if (largestPickedNum <= 20) {
                            m_pc_cloud_label[ind] = 2;  // 2 -> the label sharpest points.
                            cornerPointsSharp.push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        } else if (largestPickedNum <= 200) {
                            m_pc_cloud_label[ind] = 1;  // 1 -> the label of less sharpest points.
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        } else {
                            break;
                        }

                        m_pc_neighbor_picked[ind] = 1;

                        float times = 100;
                        // delete 5 neighbor of sharpest points.
                        for (int l = 1; l <= 5 * times; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                                break;
                            }

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5 * times; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                                break;
                            }

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = m_pc_sort_idx[k];

                    if (m_pc_neighbor_picked[ind] == 0 && m_pc_curvature[ind] < sharp_point_threshold) {
                        m_pc_cloud_label[ind] = -1;  // -1 the lable of flat points
                        surfPointsFlat.push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 5) {  // 0 the label of less flat points.
                            break;
                        }

                        m_pc_neighbor_picked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                                break;
                            }

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                                break;
                            }

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                    }
                }

                // The ublabeled point is the less flat points
                for (int k = sp; k <= ep; k++) {
                    if (m_pc_cloud_label[k] <= 0) {
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    }
                }
            }

            // voxel filter for less sharp points.
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;

            m_voxel_filter_for_surface.setInputCloud(surfPointsLessFlatScan);
            m_voxel_filter_for_surface.filter(surfPointsLessFlatScanDS);

            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }

        // printf_line;
        // printf_line;
        // printf( "sort q time %f \n", t_q_sort );
        // printf_line;
        sensor_msgs::msg::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
        laserCloudOutMsg.header.frame_id = "/camera_init";
        m_pub_laser_pc->publish(laserCloudOutMsg);

        sensor_msgs::msg::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
        cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsSharpMsg.header.frame_id = "/camera_init";
        m_pub_pc_sharp_corner->publish(cornerPointsSharpMsg);

        sensor_msgs::msg::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
        m_pub_pc_less_sharp_corner->publish(cornerPointsLessSharpMsg);

        sensor_msgs::msg::PointCloud2 surfPointsFlat2;
        pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
        surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsFlat2.header.frame_id = "/camera_init";
        m_pub_pc_surface_flat->publish(surfPointsFlat2);

        sensor_msgs::msg::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsLessFlat2.header.frame_id = "/camera_init";
        m_pub_pc_surface_less_flat->publish(surfPointsLessFlat2);

        // pub each scam
        if (m_if_pub_each_line) {
            for (int i = 0; i < m_laser_scan_number; i++) {
                sensor_msgs::msg::PointCloud2 scanMsg;
                pcl::toROSMsg(laserCloudScans[i], scanMsg);
                scanMsg.header.stamp = laserCloudMsg->header.stamp;
                scanMsg.header.frame_id = "/camera_init";
                m_pub_each_scan[i]->publish(scanMsg);
            }
        }
    }

    void init_livox_lidar_para() {
        std::string lidar_type_name;
        std::cout << "~~~~~ Init livox lidar parameters ~~~~~" << endl;
        if (this->get_parameter("lidar_type", lidar_type_name)) {
            printf("***** I get lidar_type declaration, lidar_type_name = %s ***** \r\n", lidar_type_name.c_str());

            if (lidar_type_name.compare("livox") == 0) {
                m_lidar_type = 1;
                std::cout << "Set lidar type = livox" << std::endl;
            } else {
                std::cout << "Set lidar type = velodyne" << std::endl;
                m_lidar_type = 0;
            }
        } else {
            printf("***** No lidar_type declaration ***** \r\n");
            m_lidar_type = 0;
            std::cout << "Set lidar type = velodyne" << std::endl;
        }

        if (this->get_parameter("livox_min_dis", m_livox.m_livox_min_allow_dis)) {
            std::cout << "Set livox lidar minimum distance= " << m_livox.m_livox_min_allow_dis << std::endl;
        }

        if (this->get_parameter("livox_min_sigma", m_livox.m_livox_min_sigma)) {
            std::cout << "Set livox lidar minimum sigama =  " << m_livox.m_livox_min_sigma << std::endl;
        }
        std::cout << "~~~~~ End ~~~~~" << endl;
    }
};

#endif  // LASER_FEATURE_EXTRACTION_H
