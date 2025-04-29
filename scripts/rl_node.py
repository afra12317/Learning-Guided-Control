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
        self.viz_publisher = self.create_publisher(MarkerArray, '/viz_rl', 10)

        self.laser_scan = 10 * np.ones((1, 36), dtype=np.float32)
        self.pose = np.zeros((1,3), dtype=np.float32)
        self.vels = np.zeros((1,3), dtype=np.float32)
        self.heading = np.zeros((1,2), dtype=np.float32)
        self.LF = 0.15875
        self.LR = 0.17145
        self.SCAN_INDEX = np.arange(0, 1080, 1080 // 36)
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
                                "timestep": 0.1,
                                "integrator": "rk4",
                                "control_input": ["speed", "steering_angle"],
                                "model": "st",
                                "num_beams" : 36,
                                "observation_config": {"type": "original"},
                                "params": F110Env.f1tenth_vehicle_params(),
                                "reset_config": {"type": "map_random_static"},
                                "scale": 1.0,
                            },
                            render_mode="rgb_array",
                        )
        self.N_SIM = 10 # number of future states to predict       


    def laser_callback(self, msg: LaserScan):
        min_range = msg.range_min
        max_range = msg.range_max
        values = np.array(msg.ranges)
        values[values < min_range] = min_range
        values[values > max_range] = max_range
        self.laser_scan[0] = values[self.SCAN_INDEX]

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

        self.env.reset(options={"poses" : np.array([[pos.x, pos.y, yaw]])})

        obs = {'scan' : self.laser_scan, 'pose' : self.pose, 'vel' : self.vels, 'heading' : self.heading}
        steer, vel = self.run_model(obs)
        self.heading[0,0] = steer
        self.heading[0,1] = self.get_beta(steer)
        drive = AckermannDriveStamped()
        drive.drive.speed = vel
        drive.drive.steering_angle = steer
        self.drive_publisher.publish(drive)

        ## calculate simulated positions
        xs = []
        ys = []
        yaws = []
        for _ in range(self.N_SIM):
            obs, _, _, _, _ = self.env.step(np.array([[steer, vel]]))
            scan = np.zeros((1, 36), dtype=np.float32)
            scan[0] = obs['scans'][0, self.SCAN_INDEX]
            x = float(obs['poses_x'][0])
            y = float(obs['poses_y'][0])
            yaw = float(obs['poses_theta'][0])
            vel_x = float(obs['linear_vels_x'][0])
            vel_y = float(obs['linear_vels_y'][0])
            vel_ori = float(obs['ang_vels_z'][0])
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            beta = self.get_beta(steer)
            obs = {'scan': scan,
                   'pose' : np.array([[x, y, yaw]], dtype=np.float32),
                   'vel' : np.array([[vel_x, vel_y, vel_ori]], dtype=np.float32),
                   'heading' : np.array([[steer, beta]], dtype=np.float32)}
            steer, vel = self.run_model(obs)
        self.visualize_future_pose(xs, ys, yaws)

    def run_model(self, obs):
        control = self.model.run(None, obs)
        control = control[0][0, 0]
        control = np.clip(control, -1.0, 1.0)
        control = control * self.CONTROL_MAX
        steer = float(control[0])
        vel = float(control[1])
        return steer, vel

    def get_beta(self, steer):
        return np.arctan(self.LR * np.tan(steer) / (self.LF + self.LR))

    def visualize_future_pose(self, xs, ys, yaws):
        msg = MarkerArray()
        for i, x in enumerate(xs):
            marker = Marker()
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(ys[i])
            pose_yaw = float(yaws[i])
            marker.pose.orientation.w = np.cos(pose_yaw / 2)
            marker.pose.orientation.z = np.sin(pose_yaw / 2)
            marker.color.a = marker.color.r = 1.0
            marker.color.b = marker.color.g = 0.0
            marker.header.frame_id = 'map'
            marker.ns = 'poses'
            marker.scale.x = 0.4
            marker.scale.y = 0.075
            marker.scale.z = 0.05
            marker.id = len(msg.markers)
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            msg.markers.append(marker)
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