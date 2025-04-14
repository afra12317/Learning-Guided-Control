#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import transforms3d.euler


# Pose index constants
POSE_X_IDX = 0
POSE_Y_IDX = 1
POSE_THETA_IDX = 4  # yaw index


def euler_from_quaternion(q):
    quat = [q.w, q.x, q.y, q.z]
    yaw, pitch, roll = transforms3d.euler.quat2euler(quat, axes='rzyx')
    return roll, pitch, yaw


class ObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detector_node')
        self.clip_distance = 10.0
        self.vision_width = 5.0
        self.vision_front = 10.0
        self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/obstacle_markers", 10)
        self.latest_pose = None
        self.latest_scan = None
        self.current_obstacles = np.empty((0, 2))

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("ObstacleDetectorNode started")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(ori)

        self.latest_pose = np.zeros(7)
        self.latest_pose[POSE_X_IDX] = pos.x
        self.latest_pose[POSE_Y_IDX] = pos.y
        self.latest_pose[POSE_THETA_IDX] = yaw

    def scan_callback(self, msg):
        self.latest_scan = msg

    def timer_callback(self):
        if self.latest_pose is None or self.latest_scan is None:
            return

        obstacles = self.compute_obstacle_points(self.latest_scan, self.latest_pose)
        self.current_obstacles = obstacles
        self.publish_markers(obstacles)


    def compute_obstacle_points(self, scan_msg, car_state):
        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        angles = angle_min + np.arange(len(ranges)) * angle_increment
        ranges = np.clip(ranges, 0.0, self.clip_distance)

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        in_front = (xs > 0) & (xs < self.vision_front) & (np.abs(ys) < self.vision_width / 2.0)
        xs = xs[in_front]
        ys = ys[in_front]

        theta = car_state[POSE_THETA_IDX]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        global_xs = car_state[POSE_X_IDX] + (xs * cos_theta - ys * sin_theta)
        global_ys = car_state[POSE_Y_IDX] + (xs * sin_theta + ys * cos_theta)

        return np.vstack((global_xs, global_ys)).T

    def publish_markers(self, obstacles):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obs[0])
            marker.pose.position.y = float(obs[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def get_obstacle_points(self):
        return self.current_obstacles


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()