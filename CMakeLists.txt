cmake_minimum_required(VERSION 3.5)
project(mppi_control)

foreach(PACKAGE
    ament_cmake
    ament_cmake_python
    rclcpp
    rclpy
    nav_msgs
    ackermann_msgs
    sensor_msgs
    std_msgs)
    find_package(${PACKAGE} REQUIRED)
endforeach()

install(
    DIRECTORY scripts/
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/mppi_node.py
    scripts/infer_env.py
    scripts/mppi_tracking.py
    scripts/vis_node.py
    scripts/QtMatplotlib.py
    scripts/trajectory_error_evaluator.py
    scripts/pure_pursuit_node.py
    scripts/mppi_PP.py
    scripts/obs.py
    scripts/rl_node.py
    scripts/mppi_node_og.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

ament_package()