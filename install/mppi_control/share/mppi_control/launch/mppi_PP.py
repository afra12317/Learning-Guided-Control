from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi_control',
            executable='mppi_PP.py',
            name='mppi_PP',
            output='screen'
        ),
        Node(
            package='mppi_control',
            executable='pure_pursuit_node.py',
            name='PP',
            output='screen'
        ),
    ])