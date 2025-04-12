#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from utils.ros_np_multiarray import to_numpy_f32
import numpy as np
import signal
import sys
import time

from rclpy.qos import QoSProfile, ReliabilityPolicy


class TrajectoryErrorEvaluator(Node):
    def __init__(self):
        super().__init__("trajectory_error_evaluator")
        self.reference_arr = None
        self.opt_traj_arr = None
        self.last_reference_time = 0
        self.last_opt_traj_time = 0

        self.all_errors = []  # Accumulated error values

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Float32MultiArray, "/reference_arr", self.reference_callback, qos
        )
        self.create_subscription(
            Float32MultiArray, "/opt_traj_arr", self.opt_traj_callback, qos
        )

        signal.signal(signal.SIGINT, self.signal_handler)

    def reference_callback(self, msg):
        self.reference_arr = to_numpy_f32(msg)
        self.last_reference_time = time.time()
        self.try_compare()

    def opt_traj_callback(self, msg):
        full_traj = to_numpy_f32(msg)
        # Skip first 10 points for compatibility with Visualizer_Node
        if full_traj.shape[0] > 10:
            self.opt_traj_arr = full_traj[10:]
        else:
            self.opt_traj_arr = full_traj

        self.last_opt_traj_time = time.time()
        self.try_compare()

    def try_compare(self):
        if self.reference_arr is None or self.opt_traj_arr is None:
            return

        if abs(self.last_reference_time - self.last_opt_traj_time) > 1.0:
            return

        compare_len = min(len(self.reference_arr), len(self.opt_traj_arr))
        if compare_len == 0:
            return

        ref = self.reference_arr[:compare_len, :2]
        traj = self.opt_traj_arr[:compare_len, :2]
        errors = np.linalg.norm(ref - traj, axis=1)

        avg_error = np.mean(errors)
        self.all_errors.append(avg_error)

        self.get_logger().info(
            f"Current error (compared {compare_len} points): {avg_error:.4f} meters"
        )

    def signal_handler(self, sig, frame):
        self.get_logger().info("Received shutdown signal, summarizing total error...")
        self.compute_total_average_error()
        rclpy.shutdown()
        sys.exit(0)

    def compute_total_average_error(self):
        if not self.all_errors:
            self.get_logger().info(
                "No error data collected, skipping final computation."
            )
            return

        avg = np.mean(self.all_errors)
        self.get_logger().info(
            f"Total comparisons: {len(self.all_errors)} | Average error: {avg:.4f} meters"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryErrorEvaluator()
    print(
        "Trajectory Error Evaluator Node Running. Press Ctrl+C to stop and compute final average error."
    )
    rclpy.spin(node)


if __name__ == "__main__":
    main()
