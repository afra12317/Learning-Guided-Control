#!/usr/bin/env python3
import time, os
import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
import tf_transformations

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray
from rclpy.duration import Duration

from utils.ros_np_multiarray import to_multiarray_f32
from infer_env import InferEnv
from mppi_tracking import MPPI
import utils.utils as utils
import utils.jax_utils as jax_utils
from utils.Track import Track
from utils.jax_utils import numpify
from sensor_msgs.msg import LaserScan


class MPPI_Node(Node):
    def __init__(self):
        super().__init__('mppi_node')
        self.config = utils.ConfigYAML()
        self.config.load_file('./config/config_PP.yaml')
        self.config.norm_params = np.array(self.config.norm_params).T

        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)

        map_info = np.genfromtxt(self.config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
        track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config)

        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env, jrng)

        self.control = np.asarray([0.0, 0.0])
        self.state_c_0 = np.zeros(7)
        self.pp_ref_traj = None

        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)

        # Subscriptions
        topic = "/ego_racecar/odom" if self.config.is_sim else "/pf/pose/odom"
        self.pose_sub = self.create_subscription(Odometry, topic, self.pose_callback, qos)
        self.pp_ref_sub = self.create_subscription(Float32MultiArray, "/pp_ref_traj", self.pp_ref_callback, qos)

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)
        self.lidar_scan = None

        self.get_logger().info("MPPI node initialized.")

    def pose_callback(self, pose_msg):
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist

        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = tf_transformations.euler_from_quaternion(quat)[2]

        self.state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            yaw,
            twist.angular.z,
            beta,
        ])
        
    def scan_callback(self, msg):
        self.lidar_scan = msg
    
        
    def process_lidar_to_obstacle_points(self, scan_msg, car_state):
        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        ranges = np.clip(ranges, 0.0, 10.0)  
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        in_front = (xs > 0) & (xs < 10.0) & (np.abs(ys) < 2.5)  
        xs = xs[in_front]
        ys = ys[in_front]

        theta = car_state[4]  # yaw
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        global_xs = car_state[0] + (xs * cos_theta - ys * sin_theta)
        global_ys = car_state[1] + (xs * sin_theta + ys * cos_theta)

        return np.vstack((global_xs, global_ys)).T

    def pp_ref_callback(self, msg):
        arr = np.array(msg.data, dtype=np.float32)
        if len(arr) % 2 != 0:
            return
        if self.lidar_scan is None:
            self.get_logger().warn("LiDAR data not yet received. Skipping MPPI update.")
            return
        self.pp_ref_traj = arr.reshape((-1, 2))
        reference_traj = self.interpolate_ref_from_pp(self.state_c_0.copy(), self.pp_ref_traj, self.config.n_steps)
        obs_array = self.process_lidar_to_obstacle_points(self.lidar_scan, self.state_c_0)
        self.mppi.update(jnp.asarray(self.state_c_0), jnp.asarray(reference_traj), obs_array)
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2] / 2

        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = float(mppi_control[1]) * self.config.sim_time_step + self.state_c_0[3]

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.get_logger().warn("Invalid control detected, resetting.")
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = float(self.control[0])
        drive_msg.drive.speed = float(self.control[1])
        self.drive_pub.publish(drive_msg)

        if self.reference_pub.get_subscription_count() > 0:
            ref_msg = to_multiarray_f32(numpify(reference_traj).astype(np.float32))
            self.reference_pub.publish(ref_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_msg = to_multiarray_f32(numpify(self.mppi.traj_opt).astype(np.float32))
            self.opt_traj_pub.publish(opt_msg)
        '''
        self.get_logger().info(
            f"[MPPI] Drive command: steer={self.control[0]:.3f}, speed={self.control[1]:.3f}"
        )
        '''

    def interpolate_ref_from_pp(self, state, pp_traj, n_steps):
        ref_traj = np.zeros((n_steps + 1, 3))
        max_idx = min(len(pp_traj), n_steps + 1)
        for i in range(max_idx):
            x, y = pp_traj[i]
            next_idx = min(i + 1, len(pp_traj) - 1)
            dx = pp_traj[next_idx][0] - x
            dy = pp_traj[next_idx][1] - y
            heading = np.arctan2(dy, dx) if dx != 0 or dy != 0 else state[4]
            ref_traj[i] = [x, y, heading]
        for i in range(max_idx, n_steps + 1):
            ref_traj[i] = ref_traj[max_idx - 1]
        return jnp.array(ref_traj)
    


def main(args=None):
    rclpy.init(args=args)
    node = MPPI_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()