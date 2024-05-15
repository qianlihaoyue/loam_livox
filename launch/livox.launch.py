import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_path = get_package_share_directory('loam_livox')
    config_file =  os.path.join(package_path, 'config','livox.yaml' )
    rviz_config_path = os.path.join(package_path, 'rviz_cfg', 'livox.rviz')
    
    scanRegistration_node = Node(
        package='loam_livox',
        executable='livox_scanRegistration',
        name='livox_scanRegistration',
        parameters=[config_file],
        output='screen'
    )

    laserMapping_node = Node(
        package='loam_livox',
        executable='livox_laserMapping',
        name='livox_laserMapping',
        parameters=[config_file],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path]
    )

    ld = LaunchDescription()
    ld.add_action(scanRegistration_node)
    ld.add_action(laserMapping_node)
    ld.add_action(rviz_node)

    return ld


