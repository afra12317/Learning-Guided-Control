#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import pandas as pd

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import transforms3d.euler
from geometry_msgs.msg import Point


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

        self.centerline, self.track_widths = self.load_centerline_from_csv(
            '/home/flo/lab_ws/src/Learning-Guided-Control-MPPI/waypoints/levine/centerline.csv'
        )

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("ObstacleDetectorNode started")

    def load_centerline_from_csv(self, filepath):
        df = pd.read_csv(filepath, sep=';', comment='#', header=None,
                         names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
        centerline = df[['x_m', 'y_m']].to_numpy()
        widths = df[['w_tr_right_m', 'w_tr_left_m']].to_numpy()
        return centerline, widths

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
        self.publish_detection_area()

    def convert_cartesian_to_frenet(self, point):
        diffs = self.centerline - point
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)

        closest_pt = self.centerline[min_idx]
        if min_idx < len(self.centerline) - 1:
            next_pt = self.centerline[min_idx + 1]
        else:
            next_pt = self.centerline[min_idx - 1]
        direction = next_pt - closest_pt
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        vec_to_point = point - closest_pt
        d = np.cross(direction, vec_to_point)
        s = np.sum(np.linalg.norm(np.diff(self.centerline[:min_idx+1], axis=0), axis=1))
        return s, d, min_idx

    def is_on_track(self, point):
        _, d, idx = self.convert_cartesian_to_frenet(point)
        left = min(self.track_widths[idx][1], 0.5)
        right = min(self.track_widths[idx][0], 0.5)
        safety_margin = 0.1
        if d > left or d < -right:
            return False
        if abs(d - left) < safety_margin or abs(d + right) < safety_margin:
            return False
        return True

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

        all_points = np.vstack((global_xs, global_ys)).T

        filtered_points = []
        for pt in all_points:
            if self.is_on_track(pt):
                filtered_points.append(pt)
        max_num = 50
        filtered_points = np.array(filtered_points).reshape(-1, 2)

        if len(filtered_points) > max_num:
            filtered_points = filtered_points[:max_num]
        elif len(filtered_points) < max_num:
            pad_len = max_num - len(filtered_points)
            padding = np.ones((pad_len, 2)) * 100.0
            filtered_points = np.vstack((filtered_points, padding))
        print(filtered_points)
        return np.array(filtered_points)
    

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

    def publish_detection_area(self):
        return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "valid_detection_area"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        offset = 0.5

        left_points = []
        right_points = []

        for i in range(len(self.centerline) - 1):
            pt = self.centerline[i]
            pt_next = self.centerline[i + 1]
            direction = pt_next - pt
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            normal = np.array([-direction[1], direction[0]])
            left_pt = pt + offset * normal
            right_pt = pt - offset * normal
            left_points.append(left_pt)
            right_points.append(right_pt)

        strip = np.vstack((left_points, right_points[::-1]))
        for p in strip:
            marker.points.append(Point(x=p[0], y=p[1], z=0.0))

        self.marker_pub.publish(MarkerArray(markers=[marker]))

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