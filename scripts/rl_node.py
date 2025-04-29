#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import onnxruntime as ort
import numpy as np
from f1tenth_gym.envs.f110_env import F110Env, Track
import gymnasium as gym
import pathlib

class RLNode(Node):
    
    def __init__(self):
        super().__init__('rl_node')
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        ## publisher for visualizing predicted t+1 pose
        self.viz_publisher = self.create_publisher(Marker, '/viz_rl', 10)

        self.laser_scan = 10 * np.ones((1, 36), dtype=np.float32)
        self.pose = np.zeros((1,3), dtype=np.float32)
        self.vels = np.zeros((1,3), dtype=np.float32)
        self.heading = np.zeros((1,2), dtype=np.float32)
        self.LF = 0.15875
        self.LR = 0.17145
        self.model = ort.InferenceSession('/home/ubuntu/ese6150_ws/src/Learning-Guided-Control-MPPI/rl_models/out.onnx')
        self.CONTROL_MAX = np.array([0.4189, 5.0])
        # create an environment backend for simulating actions to predict future states and lidar scans
        path = '/home/ubuntu/f1final/f1tenth_gym/maps/levine/levine.yaml'
        path = pathlib.Path(path)
        loaded_map = Track.from_track_path(path)
        self.env = gym.make(
                            "f1tenth_gym:f1tenth-v0",
                            config={
                                "map": loaded_map,
                                "num_agents": 1,
                                "timestep": 0.01,
                                "integrator": "rk4",
                                "control_input": ["speed", "steering_angle"],
                                "model": "st",
                                "observation_config": {"type": "original"},
                                "params": F110Env.f1fifth_vehicle_params(),
                                "reset_config": {"type": "map_random_static"},
                                "scale": 1.0,
                            },
                            render_mode="rgb_array",
                        )

        # self.env_reset = False
        


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

        # if not self.env_reset:
            # self.env_reset = True
        self.env.reset(options={"poses" : np.array([[pos.x, pos.y, yaw]])})

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
        self.heading[0,0] = steer
        self.heading[0,1] = np.arctan(self.LR * np.tan(steer) / (self.LF + self.LR))
        drive = AckermannDriveStamped()
        drive.drive.speed = vel
        drive.drive.steering_angle = steer

        ## calculate simulated positions
        obs, _, _, _, _ = self.env.step(np.array([[steer, vel]]))
        # print(obs)
        self.visualize_future_pose(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])

        self.drive_publisher.publish(drive)

    def visualize_future_pose(self, pose_x, pose_y, pose_yaw):
        msg = Marker()
        msg.pose.position.x = float(pose_x)
        msg.pose.position.y = float(pose_y)
        pose_yaw = float(pose_yaw)
        msg.pose.orientation.w = np.cos(pose_yaw / 2)
        msg.pose.orientation.z = np.sin(pose_yaw / 2)
        msg.color.a = msg.color.r = 1.0
        msg.color.b = msg.color.g = 0.0
        msg.header.frame_id = 'map'
        msg.ns = 'poses'
        msg.scale.x = 0.4
        msg.scale.y = 0.075
        msg.scale.z = 0.05
        msg.id = 0
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        self.viz_publisher.publish(msg)



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