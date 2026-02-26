# This is the launch file that starts up the basic QBot Platform nodes,
# plus the TF node. Then start the cartographer.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (PathJoinSubstitution, LaunchConfiguration)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import launch_ros.actions


DeclareLaunchArgument('map_file')
DeclareLaunchArgument('use_sim_time')
slam_config_path = PathJoinSubstitution([
        FindPackageShare('qcar2_nodes'),
        'launch',                 # Folder where setup.py installed it
        'async_slam_param.yaml'   # The filename
    ])

# Path to the nav2_bringup package (where the standard launch files are)
nav2_bringup_dir = get_package_share_directory('nav2_bringup')
# Set the default path to your params file (change 'config/nav2_params.yaml')
# default_params_path = PathJoinSubstitution([pkg_share, 'config', 'nav2_params.yaml'])
default_nav2_params = os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml')

def generate_launch_description():

    

   
    qcar2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('qcar2_nodes'), 'launch', 'qcar2_virtual_launch.py')]
        )
    )


    qcar2_nav2_converter = Node(
    package='qcar2_nodes',
    executable='nav2_qcar2_converter',
    name='nav2_qcar2_converter',
    parameters=[{
            "use_sim_time":True,
        }],
    )


    path_pub_node= Node(
        package ='qcar2_autonomy',
        executable ='path_pub',
        name ='path_pub',
        # parameters=[{"use_sim_time":True}]
    )
    odom_publisher= Node(
        package ='qcar2_autonomy',
        executable ='odom_pub',
        name ='odom_pub',
        # parameters=[{"use_sim_time":True}]
    )


    imu_odom_node=Node(
        package='qcar2_autonomy',
        executable='imu_odom',
        name='imu_odom'


    )


    ekf_node=Node(
        package="qcar2_cpp",
        executable='ekf_node',
        name='ekf_node'
    )

    slam_node = Node(
    package='slam_toolbox',
    executable='async_slam_toolbox_node',
    name='slam_node',
    output='screen',
    parameters=[{
        # --- Time ---
        'use_sim_time': True,

        # --- Frames (CRITICAL) ---
        'map_frame': 'map',
        'odom_frame': 'odom',
        'base_frame': 'base_link',
        'tracking_frame': 'base_scan',


        # --- Map ---
        'map_file_name': 'ORIGIN_MAP',

        # --- Topics ---
        'scan_topic': 'scan',
        'mode': 'mapping',

        # --- TF Stability (VERY IMPORTANT) ---
        'lookup_transform_timeout_sec': 0.8,
        'tf_buffer_duration': 30.0,

        # --- Queue & Rate Control (FIXES YOUR ERROR) ---
        'scan_queue_size': 20,
        'throttle_scans': 1,

        # --- Sensor Limits ---
        'max_laser_range': 10.0,

        # --- Motion thresholds ---
        'minimum_travel_distance': 0.05,
        'minimum_travel_heading': 0.05,
       
    }]
)
    
    # slam_localization_node = Node(
    #     package='slam_toolbox',
    #     executable='localization_slam_toolbox_node',
    #     name='slam_toolbox',
    #     output='screen',
    #     parameters=[{
    #         # --- Mode ---
    #         'use_sim_time': True,
    #         'mode': 'localization',

    #         # --- Map ---
    #         'map_file_name': 'maps/ORIGIN_MAP',

    #         # --- Frames ---
    #         'map_frame': 'map',
    #         'odom_frame': 'odom',
    #         'base_frame': 'base_link',
    #         'tracking_frame': 'base_scan',

    #         # --- Scan ---
    #         'scan_topic': 'scan',

    #         # --- TF tuning ---
    #         'lookup_transform_timeout_sec': 0.5,
    #         'tf_buffer_duration': 30.0,
    #         'map_start_pose':[-1.28700493, -0.61549585,-0.7802] ,      #[-1.205, -0.83, -0.7802],  


    #         # --- Performance ---
    #         'throttle_scans': 1,
    #         'scan_queue_size': 20,
    #         'max_laser_range': 10.0,

    #         # --- Localization tuning ---
    #         'minimum_travel_distance': 0.0,
    #         'minimum_travel_heading': 0.0
    #     }]
    # )


    slam_localization=LaunchDescription([
        launch_ros.actions.Node(
          parameters=[
            'Parameters/slam_localization_params.yaml'
          ],
          package='slam_toolbox',
          executable='localization_slam_toolbox_node',
          name='slam_toolbox',
          output='screen'
        )
    ])


    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', 'rviz/mpc.rviz'], # Optional: Load a saved config
        parameters=[{'use_sim_time': True}]    
    )


    mux_node=Node(
            package='twist_mux',
            executable='twist_mux',
            output='screen',
            parameters=[
               'Parameters/twist_mux_locks.yaml',
                'Parameters/twist_mux_topics.yaml'],
            remappings=[
                # The mux listens to input topics defined in YAML.
                # It outputs to 'cmd_vel_out' by default.
                # We remap this output to the standard '/cmd_vel' 
                # that your motor controller listens to.
                ('cmd_vel_out', '/cmd_vel_nav')
            ]
        )






    



    return LaunchDescription([
        rviz_node,
        qcar2_launch,
        qcar2_nav2_converter,
        imu_odom_node,
        ekf_node,
        # odom_publisher,
        # start_localization_cmd,
        # slam_node,
        slam_localization,
        path_pub_node,
        mux_node,
        
    ])
