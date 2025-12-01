from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav_package_cw',
            executable='nav_node_cw',
            name='navigator',
            output='screen'
        ),
        Node(
            package='eye_package_cw',
            executable='eye_node_cw',
            name='eye',
            output='screen'
        ),
        Node(
            package='frontier_package_cw',
            executable='frontier_node_cw',
            name='frontier',
            output='screen'
        )
    ])