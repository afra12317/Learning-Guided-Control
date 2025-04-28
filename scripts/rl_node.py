#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import onnxruntime as ort
import numpy as np

class RLNode(Node):
    
    def __init__(self):
        super().__init__('rl_node')
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.laser_scan = 10 * np.ones((1, 36), dtype=np.float32)
        self.pose = np.zeros((1,3), dtype=np.float32)
        self.vels = np.zeros((1,3), dtype=np.float32)
        self.heading = np.zeros((1,2), dtype=np.float32)
        self.LF = 0.15875
        self.LR = 0.17145
        self.model = ort.InferenceSession('/home/ubuntu/ese6150_ws/src/Learning-Guided-Control-MPPI/rl_models/out.onnx')
        self.CONTROL_MAX = np.array([0.4189, 5.0])
        # print('NODE MADE')


    def laser_callback(self, msg: LaserScan):
        min_range = msg.range_min
        max_range = msg.range_max
        values = np.array(msg.ranges)
        values[values < min_range] = min_range
        values[values > max_range] = max_range
        angle_increments = np.arange(0, len(values), 1080 // 36)
        self.laser_scan[0] = values[angle_increments]
        # print(self.laser_scan.shape)

    def pose_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        lin_vel = msg.twist.twist.linear
        ori_vel = msg.twist.twist.angular
        yaw = 2 * np.arccos(ori.w)
        self.pose[0,0] = pos.x
        self.pose[0,1] = pos.y
        self.pose[0,2] = yaw
        self.vels[0,0] = lin_vel.x
        self.vels[0,1] = lin_vel.y
        self.vels[0,2] = ori_vel.z

        obs = {'scan' : self.laser_scan, 'pose' : self.pose, 'vel' : self.vels, 'heading' : self.heading}
        control = self.model.run(None, obs)
        # print(control[0][0, 0])
        control = control[0][0, 0]
        control = np.clip(control, -1.0, 1.0)
        control = control * self.CONTROL_MAX
        # print(control)
        # print(control)s
        steer = float(control[0])
        vel = float(control[1])
        # print(type(steer), vel)
        self.heading[0,0] = -steer
        self.heading[0,1] = np.arctan(self.LR * np.tan(steer) / (self.LF + self.LR))
        drive = AckermannDriveStamped()
        drive.drive.speed = vel
        drive.drive.steering_angle = steer
        self.drive_publisher.publish(drive)

def main(args=None):
    rclpy.init(args=args)
    rl_node = RLNode()

    try:
        rclpy.spin(rl_node)
    finally:
        rl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()