from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi_control',
            executable='mppi_node.py',
            name='mppi_node',
            output='screen'
        ),
        Node(
            package='mppi_control',
            executable='vis_node.py',
            name='vis_node',
            output='screen'
        ),
        Node(
            package='mppi_control',
            executable='trajectory_error_evaluator.py',
            name='trajectory_error_evaluator',
            output='screen'
        ),
    ])