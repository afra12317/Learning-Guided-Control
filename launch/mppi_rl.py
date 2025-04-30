from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

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
        # rl_node using NumPy 1.24 virtualenv
        ExecuteProcess(
            cmd=[
                "/home/ubuntu/venvs/numpy124/bin/python",
                "/home/ubuntu/ese6150_ws/src/Learning-Guided-Control-MPPI/scripts/rl_node.py"
            ],
            name='rl_node',
            output='screen'
        )
    ])
    
'''
        Node(
            package='mppi_control',
            executable='trajectory_error_evaluator.py',
            name='trajectory_error_evaluator',
            output='screen'
        ),
'''