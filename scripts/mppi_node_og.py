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

import cProfile
import pstats
import atexit
import pandas as pd
from visualization_msgs.msg import Marker, MarkerArray
from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from scipy.spatial import cKDTree



def transform_lidar_points(
    ranges, angles, car_x, car_y, theta
):
    xs = ranges * jnp.cos(angles)
    ys = ranges * jnp.sin(angles)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    global_xs = car_x + (xs * cos_theta - ys * sin_theta)
    global_ys = car_y + (xs * sin_theta + ys * cos_theta)
    return jnp.stack((global_xs, global_ys), axis=-1)   


class MPPI_Node(Node):
    def __init__(self):
        super().__init__("mppi_node")
        self.config = utils.ConfigYAML()
        self.config.load_file(
            "/home/nvidia/f1tenth_ws/src/Learning-Guided-Control-MPPI/config/config_example.yaml"
        )
        self.config.norm_params = np.array(self.config.norm_params).T

        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)

        map_info = np.genfromtxt(
            self.config.map_dir + "map_info.txt", delimiter="|", dtype="str"
        )
        track, self.config = Track.load_map(
            self.config.map_dir, map_info, self.config.map_ind, self.config
        )

        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env, jrng, track=track)
        self.centerline, self.track_widths = self.load_centerline_from_csv(
            self.config.map_dir + "levine/centerline.csv"
        )

        self.control = np.asarray([0.0, 0.0])
        self.lidar_scan = None
        self.track = track
        # Dummy init
        state_c_0 = np.zeros(7)
        reference_traj, _ = self.infer_env.get_refernece_traj_jax(
            state_c_0.copy(), self.config.ref_vel, self.config.n_steps
        )
        self.mppi.update(
            jnp.asarray(state_c_0), jnp.asarray(reference_traj), np.zeros((0, 2))
        )

        self.get_logger().info("MPPI with obstacle avoidance initialized")

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        topic = "/ego_racecar/odom" if self.config.is_sim else "/pf/pose/odom"
        self.pose_sub = self.create_subscription(
            Odometry, topic, self.pose_callback, qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(
            Float32MultiArray, "/reference_arr", qos
        )
        self.opt_traj_pub = self.create_publisher(
            Float32MultiArray, "/opt_traj_arr", qos
        )
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, "/mppi/obstacle_markers", qos
        )
        self.center_kdtree = cKDTree(self.centerline)

    def load_centerline_from_csv(self, filepath):
        df = pd.read_csv(
            filepath,
            sep=";",
            comment="#",
            header=None,
            names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"],
        )
        centerline = df[["x_m", "y_m"]].to_numpy()
        widths = df[["w_tr_right_m", "w_tr_left_m"]].to_numpy()
        return centerline, widths

    def scan_callback(self, msg):
        self.lidar_scan = msg

    def convert_cartesian_to_frenet_np(self, point):
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
        s = np.sum(
            np.linalg.norm(np.diff(self.centerline[: min_idx + 1], axis=0), axis=1)
        )
        return s, d, min_idx

    def batch_is_obs_on_track(self, points):
        dists, min_idx = self.center_kdtree.query(points, k=1)

        closest_pts = self.centerline[min_idx]
        next_idx = np.clip(min_idx + 1, 0, len(self.centerline) - 1)
        next_pts = self.centerline[next_idx]
        direction = next_pts - closest_pts
        norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
        direction = direction / norm

        vec_to_points = points - closest_pts
        d = np.cross(direction, vec_to_points)

        lefts = np.minimum(self.track_widths[min_idx][:, 1], self.config.left)
        rights = np.minimum(self.track_widths[min_idx][:, 0], self.config.right)

        margin = self.config.safety_margin
        mask = (
            (d < lefts)
            & (d > -rights)
            & (np.abs(d - lefts) > margin)
            & (np.abs(d + rights) > margin)
        )
        return mask
    
    def filter_and_pad_obstacles(self, scan_msg, car_state):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        mask_range = np.isfinite(ranges) & (ranges > 0.05) & (ranges < 5.0)
        idx = np.nonzero(mask_range)[0]

        if idx.size == 0:
            print("No valid LiDAR points.")
            return np.full((self.config.max_obs, 2), self.config.invalid_pos, dtype=np.float32)

        angles = scan_msg.angle_min + scan_msg.angle_increment * idx
        filtered_ranges = ranges[idx]
        xs_local = filtered_ranges * np.cos(angles)
        front_mask = xs_local > 0.0
        if np.count_nonzero(front_mask) == 0:
            print("No points detected in front of the vehicle!")
            return np.full((self.config.max_obs, 2), self.config.invalid_pos, dtype=np.float32)

        filtered_ranges = filtered_ranges[front_mask]
        angles = angles[front_mask]
        obs_world_jax = transform_lidar_points(
            jnp.array(filtered_ranges),
            jnp.array(angles),
            car_state[0],
            car_state[1],
            car_state[4]
        )
        obs_world = np.array(obs_world_jax, dtype=np.float32)
        yaws = np.full((angles.shape[0],), 0, dtype=np.float32)
        obs_with_yaw = np.concatenate([obs_world, yaws[:, None]], axis=-1)
        frenet = self.track.vmap_cartesian_to_frenet_jax(jnp.array(obs_with_yaw))

        d = frenet[:, 1] 
    
        mask = (
            abs(d) < 0.5
        )
        obs_valid = obs_world[mask]
        if len(obs_valid) == 0:
            print("No obstacles within track boundary (3.5m)!")
        max_obs = self.config.max_obs
        invalid_pos = jnp.array(self.config.invalid_pos)
        num_valid = int(obs_valid.shape[0])
        num_pad = max(0, max_obs - num_valid)

        if num_pad > 0:
            pad = jnp.tile(invalid_pos[None, :], (num_pad, 1))
            obs_valid = jnp.concatenate([obs_valid, pad], axis=0)
        return obs_valid[:max_obs]
    
    def moving_average_filter(self, data, window_size=3):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    def process_lidar_to_obstacle_points(self, scan_msg, car_state):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        smoothed_ranges = self.moving_average_filter(ranges, window_size=5)
        valid_mask = np.isfinite(smoothed_ranges) & (smoothed_ranges > 0.05) & (smoothed_ranges < 7.0)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            return np.full(
                (self.config.max_obs, 2), self.config.invalid_pos, dtype=np.float32
            )

        step = 3
        downsampled_indices = valid_indices[::step]
        angles = angle_min + angle_increment * downsampled_indices
        sampled_ranges = smoothed_ranges[downsampled_indices]

        raw_obs_points = transform_lidar_points(
        jnp.array(sampled_ranges),
        jnp.array(angles),
        car_state[0],
        car_state[1],
        car_state[4]
        )
        raw_obs_points = np.array(raw_obs_points)
        mask = self.batch_is_obs_on_track(raw_obs_points)
        on_track_points = raw_obs_points[mask]
        result = np.full(
            (self.config.max_obs, 2), self.config.invalid_pos, dtype=np.float32
        )
        n = min(on_track_points.shape[0], self.config.max_obs)
        result[:n, :] = on_track_points[:n]
        return result

    def pose_callback(self, pose_msg):
        if self.lidar_scan is None:
            self.get_logger().warn("LiDAR scan not received yet.")
            return

        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        yaw = tf_transformations.euler_from_quaternion(quat)[2]

        state_c_0 = np.array(
            [
                pose.position.x,
                pose.position.y,
                self.control[0],
                max(twist.linear.x, self.config.init_vel),
                yaw,
                twist.angular.z,
                beta,
            ]
        )

        obs_points = self.process_lidar_to_obstacle_points(self.lidar_scan, state_c_0)
        #obs_points = self.filter_and_pad_obstacles(self.lidar_scan, state_c_0)
        reference_traj, _ = self.infer_env.get_refernece_traj_jax(
            state_c_0.copy(),
            max(twist.linear.x, self.config.ref_vel),
            self.config.n_steps,
        )

        self.mppi.update(
            jnp.asarray(state_c_0), jnp.asarray(reference_traj), obs_points
        )
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2] / 2
        self.control[0] = (
            float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        )
        self.control[1] = (
            float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
        )
        
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)
        '''
        if twist.linear.x < self.config.init_vel:
            self.control = [0.0, self.config.init_vel * 2]
        '''
        
        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)
            

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        drive_msg.drive.speed = max(drive_msg.drive.speed, 3.0)
        #drive_msg.drive.speed = min(drive_msg.drive.speed, 0.0)
        self.drive_pub.publish(drive_msg)
        #self.get_logger().info(f"velocity: {drive_msg.drive.speed}")
        #self.publish_obstacle_markers(obs_points)

        
            
            

    def publish_obstacle_markers(self, obstacles):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, obs in enumerate(obstacles):
            if np.any(np.isclose(obs, self.config.invalid_pos)):
                continue

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
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = int(1e8)
            marker_array.markers.append(marker)

        self.obstacle_marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    """
    profiler = cProfile.Profile()
    profiler.enable()

    def shutdown_profiler():
        profiler.disable()
        profiler.dump_stats("mppi_profile.prof")
        print("[Profiler] Saved to mppi_profile.prof")
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)  

    atexit.register(shutdown_profiler)
    """
    try:
        rclpy.spin(mppi_node)
    finally:
        mppi_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()