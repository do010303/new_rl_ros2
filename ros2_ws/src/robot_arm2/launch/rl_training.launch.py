#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('robot_arm2').find('robot_arm2')
    
    # Set Gazebo resource path to find custom models
    models_path = os.path.join(pkg_share, 'models')
    worlds_path = os.path.join(pkg_share, 'worlds')
    
    # Get existing GZ_SIM_RESOURCE_PATH or empty string
    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    
    # Append our paths
    if gz_resource_path:
        new_gz_resource_path = f"{gz_resource_path}:{models_path}:{pkg_share}"
    else:
        new_gz_resource_path = f"{models_path}:{pkg_share}"
    
    # Set environment variable
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=new_gz_resource_path
    )
    
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("robot_arm2"),
                    "urdf",
                    "New.xacro",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Gazebo Fortress with RL training world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([
                    FindPackageShare('robot_arm2'),
                    'worlds',
                    'rl_training.world'
                ]),
                ' -r'
            ]
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot_arm',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # Joint State Broadcaster Spawner (delayed to ensure Gazebo plugin loads)
    joint_state_broadcaster_spawner = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )

    # Arm Controller Spawner (delayed)
    arm_controller_spawner = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_controller'],
                output='screen'
            )
        ]
    )
    
    # Bridge for Gazebo services (entity spawn/delete for target sphere)
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/default/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/default/remove@ros_gz_interfaces/srv/DeleteEntity'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_gz_resource_path,  # Set environment variable first
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        gz_bridge,  # Bridge for entity spawn/delete
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
    ])
