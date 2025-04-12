#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
import csv

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray
import transforms3d.euler


def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = np.sum(diffs ** 2, axis=1)
    dots = np.array([np.dot(point - trajectory[i], diffs[i]) for i in range(len(diffs))])
    t = dots / l2s
    t = np.clip(t, 0.0, 1.0)
    projections = trajectory[:-1] + (t[:, np.newaxis] * diffs)
    dists = np.linalg.norm(projections - point, axis=1)
    idx = np.argmin(dists)
    return projections[idx], dists[idx], t[idx], idx


def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, start_idx):
    for i in range(int(start_idx), len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]
        d = end - start
        f = start - point

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            continue
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        for t in [t1, t2]:
            if 0 <= t <= 1:
                return start + t * d, i, t
    return None, None, None


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.lookahead_distance = 1.0  # meters
        self.last_goal_point = None

        # === Set path to your waypoints file ===
        self.waypoints = self.load_waypoints(
            '/home/flo/lab_ws/src/Learning-Guided-Control-MPPI/waypoints/levine/levine_2.csv'
        )

        # === Subscriptions ===
        self.odom_sub = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10)

        # === Publishers ===
        self.traj_pub = self.create_publisher(Float32MultiArray, "/pp_ref_traj", 10)
        self.waypoints_pub = self.create_publisher(PoseArray, "/waypoints", 10)

        # Timer for visualizing waypoints
        self.timer = self.create_timer(0.2, self.timer_callback)

    def load_waypoints(self, path):
        waypoints = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                try:
                    x, y = float(row[1]), float(row[2])
                    waypoints.append((x, y))
                except:
                    continue
        return np.array(waypoints)

    def odom_callback(self, msg):
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.last_position = np.array([x, y])

            _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
            position = np.array([x, y])

            goal, idx = self.find_goal_point(position)
            if idx is not None:
                self.publish_pp_ref_traj(idx, num_points=20)
                #self.get_logger().info(f"[PP] Publish traj from idx={idx}, pt={self.waypoints[idx]}")
            else:
                pass
                #self.get_logger().warn("[PP] No valid goal found")

        except Exception as e:
            self.get_logger().warn(f"PurePursuit error: {e}")

    def find_goal_point(self, position):
        trajectory = self.waypoints
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, trajectory)

        if nearest_dist < self.lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, self.lookahead_distance, trajectory, i + t
            )
            if lookahead_point is not None:
                return lookahead_point, i2
        return None, None

    def publish_pp_ref_traj(self, start_idx, num_points=10):
        if start_idx is None:
            return
        traj = self.waypoints[start_idx:start_idx + num_points]
        if len(traj) == 0:
            return
        arr = Float32MultiArray()
        arr.data = traj.flatten().tolist()  # [x0, y0, x1, y1, ...]
        self.traj_pub.publish(arr)

    def euler_from_quaternion(self, q):
        quat = [q.w, q.x, q.y, q.z]
        return transforms3d.euler.quat2euler(quat)

    def timer_callback(self):
        pa = PoseArray()
        pa.header.frame_id = "map"
        pa.header.stamp = self.get_clock().now().to_msg()
        for wp in self.waypoints:
            pose = Pose()
            pose.position.x = float(wp[0])
            pose.position.y = float(wp[1])
            pa.poses.append(pose)
        self.waypoints_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()