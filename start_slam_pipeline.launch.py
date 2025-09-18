import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory for your custom package
    pkg_share_dir = get_package_share_directory('my_ekf_py_package')
    
    # Path to the SLAM Toolbox configuration file
    slam_config_file = os.path.join(pkg_share_dir, 'config', 'slam_config.yaml')

    # 1. Launch the CARLA ROS Bridge
    # This assumes you have the carla_ros_bridge package installed and its executable is named 'carla_ros_bridge'
    carla_bridge_node = Node(
        package='carla_ros_bridge',
        executable='carla_ros_bridge',
        name='carla_ros_bridge',
        output='screen'
        # Add any necessary parameters for the bridge here if needed
        # E.g., parameter_file=some_config_file
    )

    # 2. Launch your custom ES-EKF Node
    # This node subscribes to IMU/GNSS and publishes fused odometry
    es_ekf_node = Node(
        package='my_ekf_py_package',
        executable='es_ekf_node',
        name='es_ekf_node',
        output='screen',
        remappings=[
            ('/carla/ego_vehicle/imu', '/carla/ego_vehicle/imu'),
            ('/carla/ego_vehicle/gnss', '/carla/ego_vehicle/gnss'),
            # The output topic from your EKF node
            ('/odometry/filtered', '/odometry/filtered')
        ]
    )

    # 3. Launch SLAM Toolbox
    # This node takes the EKF odometry and LiDAR scans to build the map
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_config_file
        ]
    )

    return LaunchDescription([
        carla_bridge_node,
        es_ekf_node,
        slam_toolbox_node
    ])