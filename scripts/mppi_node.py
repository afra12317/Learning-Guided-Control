#!/usr/bin/env python3
import time, os, sys
import numpy as np
import jax
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from infer_env import InferEnv
from mppi_tracking import MPPI
import utils.utils as utils
from utils.jax_utils import numpify
import utils.jax_utils as jax_utils
from utils.Track import Track

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('mppi_node')
        self.config = utils.ConfigYAML()
        self.config.load_file('./config/config_example.yaml')
        self.config.norm_params = np.array(self.config.norm_params).T

        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)    

        map_info = np.genfromtxt(self.config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
        track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config)

        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env, jrng)

        self.control = np.asarray([0.0, 0.0])
        self.lidar_scan = None

        # Dummy init
        state_c_0 = np.zeros(7)
        reference_traj, _ = self.infer_env.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj), np.zeros((0, 2)))

        self.get_logger().info("MPPI with obstacle avoidance initialized")

        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST, depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)

        topic = "/ego_racecar/odom" if self.config.is_sim else "/pf/pose/odom"
        self.pose_sub = self.create_subscription(Odometry, topic, self.pose_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)

    def scan_callback(self, msg):
        self.lidar_scan = msg

    def process_lidar_to_obstacle_points(self, scan_msg, car_state):
        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        ranges = np.clip(ranges, 0.0, 10.0)  # Limit range to avoid noise

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        in_front = (xs > 0) & (xs < 10.0) & (np.abs(ys) < 2.5)
        xs = xs[in_front]
        ys = ys[in_front]

        theta = car_state[4]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        global_xs = car_state[0] + (xs * cos_theta - ys * sin_theta)
        global_ys = car_state[1] + (xs * sin_theta + ys * cos_theta)

        return np.vstack((global_xs, global_ys)).T

    def pose_callback(self, pose_msg):
        if self.lidar_scan is None:
            self.get_logger().warn("LiDAR scan not received yet.")
            return

        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = tf_transformations.euler_from_quaternion(quat)[2]

        state_c_0 = np.array([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            yaw,
            twist.angular.z,
            beta
        ])

        obs_points = self.process_lidar_to_obstacle_points(self.lidar_scan, state_c_0)
        reference_traj, _ = self.infer_env.get_refernece_traj(state_c_0.copy(), max(twist.linear.x, self.config.ref_vel), self.config.n_steps)

        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj), obs_points)
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2] / 2
        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.get_logger().warn("Invalid control. Resetting.")
            self.control = np.array([0.0, self.config.init_vel])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        self.drive_pub.publish(drive_msg)

        if self.reference_pub.get_subscription_count() > 0:
            self.reference_pub.publish(to_multiarray_f32(numpify(reference_traj).astype(np.float32)))

        if self.opt_traj_pub.get_subscription_count() > 0:
            self.opt_traj_pub.publish(to_multiarray_f32(numpify(self.mppi.traj_opt).astype(np.float32)))

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)
    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()