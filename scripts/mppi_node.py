#!/usr/bin/env python3
import time, os, sys
import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
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
        super().__init__("lmppi_node")
        self.config = utils.ConfigYAML()
        self.config.load_file("/home/ubuntu/f1tenth_ws/src/Learning-Guided-Control-MPPI/config/config_example.yaml")
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
        self.mppi = MPPI(self.config, self.infer_env, jrng)

        self.control = np.asarray([0.0, 0.0])
        self.reference_traj = np.zeros((self.config.n_steps + 1, 7), dtype=np.float32)
        self.control_seq = np.zeros((self.config.n_steps, 2), dtype=np.float32)

        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(self.reference_traj))
        self.get_logger().info("MPPI initialized with RL reference trajectory.")

        self.hz = []

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        # Subscribers
        if self.config.is_sim:
            topic = "/ego_racecar/odom"
        else:
            topic = "/pf/pose/odom"
        self.pose_sub = self.create_subscription(
            Odometry, topic, self.pose_callback, qos
        )
        self.reference_sub = self.create_subscription(
            Float32MultiArray, "/rl/ref_traj", self.reference_callback, qos
        )

        self.control_seq_sub = self.create_subscription(
            Float32MultiArray, "/rl/control_seq", self.control_seq_callback, qos
        )

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(
            Float32MultiArray, "/reference_arr", qos
        )
        self.opt_traj_pub = self.create_publisher(
            Float32MultiArray, "/opt_traj_arr", qos
        )

    def reference_callback(self, msg: Float32MultiArray):
        self.reference_traj = to_numpy_f32(msg)

    def control_seq_callback(self, msg: Float32MultiArray):
        arr = to_numpy_f32(msg)
        try:
            self.control_seq = arr.reshape(self.config.n_steps, 2)
        except ValueError:
            self.get_logger().error(
                f"Control sequence shape mismatch: {arr.shape} != ({self.config.n_steps}, 2)"
            )
            return

    def pose_callback(self, pose_msg):
        t1 = time.time()
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist

        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        theta = tf_transformations.euler_from_quaternion(quaternion)[2]

        state_c_0 = np.asarray(
            [
                pose.position.x,
                pose.position.y,
                self.control[0],
                max(twist.linear.x, self.config.init_vel),
                theta,
                twist.angular.z,
                beta,
            ]
        )

        reference_traj = self.reference_traj.copy()
        self.mppi.update(
            jnp.asarray(state_c_0),
            jnp.asarray(reference_traj),
            jnp.asarray(self.control_seq),
        )

        mppi_control = numpify(self.mppi.a_opt[0])
        self.control[0] = float(mppi_control[0])
        self.control[1] = float(mppi_control[1])

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = min(max(self.control[1], 2.0), 3.0)
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPPI_Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
