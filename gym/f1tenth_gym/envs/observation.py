from __future__ import annotations
from abc import abstractmethod
from typing import List

import gymnasium as gym
import numpy as np


class Observation:
    """
    Abstract class for observations. Each observation must implement the space and observe methods.

    :param env: The environment.
    :param vehicle_id: The id of the observer vehicle.
    :param kwargs: Additional arguments.
    """

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()


class OriginalObservation(Observation):
    def __init__(self, env):
        super().__init__(env)

    def space(self):
        num_agents = self.env.unwrapped.num_agents
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )  # add 1.0 to avoid small errors
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_range,
                    shape=(num_agents, scan_size),
                    dtype=np.float32,
                ),
                "poses_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_theta": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "ang_vels_z": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "collisions": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
                ),
                "lap_times": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "lap_counts": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
            }
        )

        return obs_space

    def observe(self):
        # state indices
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(
            7
        )  # 7 largest state size (ST Model)

        observations = {
            "ego_idx": self.env.unwrapped.sim.ego_idx,
            "scans": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "linear_vels_x": [],
            "linear_vels_y": [],
            "ang_vels_z": [],
            "collisions": [],
            "lap_times": [],
            "lap_counts": [],
        }

        for i, agent in enumerate(self.env.unwrapped.sim.agents):
            agent_scan = self.env.unwrapped.sim.agent_scans[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]
            collision = self.env.unwrapped.sim.collisions[i]

            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]

            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            observations["scans"].append(agent_scan)
            observations["poses_x"].append(x)
            observations["poses_y"].append(y)
            observations["poses_theta"].append(theta)
            observations["linear_vels_x"].append(vx)
            observations["linear_vels_y"].append(vy)
            observations["ang_vels_z"].append(angvel)
            observations["collisions"].append(collision)
            observations["lap_times"].append(lap_time)
            observations["lap_counts"].append(lap_count)

        # cast to match observation space
        for key in observations.keys():
            if isinstance(observations[key], np.ndarray) or isinstance(
                observations[key], list
            ):
                observations[key] = np.array(observations[key], dtype=np.float32)

        return observations


class FeaturesObservation(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        complete_space = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict = {
                "scan": gym.spaces.Box(
                    low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32
                ),
                "pose_x": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "pose_y": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "pose_theta": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "linear_vel_x": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "linear_vel_y": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "ang_vel_z": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "delta": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "beta": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(), dtype=np.float32
                ),
                "collision": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
                "lap_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "lap_count": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
            complete_space[agent_id] = gym.spaces.Dict(
                {k: agent_dict[k] for k in self.features}
            )

        obs_space = gym.spaces.Dict(complete_space)
        return obs_space

    def observe(self):
        obs = {}  # dictionary agent_id -> observation dict

        for i, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = self.env.unwrapped.sim.agent_scans[i]
            agent = self.env.unwrapped.sim.agents[i]

            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]
            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            # create agent's observation dict
            agent_obs = {
                "scan": scan,
                "pose_x": x,
                "pose_y": y,
                "pose_theta": theta,
                "linear_vel_x": vx,
                "linear_vel_y": vy,
                "ang_vel_z": angvel,
                "delta": delta,
                "beta": beta
            }

            # add agent's observation to multi-agent observation
            obs[agent_id] = {k: agent_obs[k] for k in self.features}

            # cast to match observation space
            for key in obs[agent_id].keys():
                if (
                    isinstance(obs[agent_id][key], np.ndarray)
                    or isinstance(obs[agent_id][key], list)
                    or isinstance(obs[agent_id][key], float)
                ):
                    obs[agent_id][key] = np.array(obs[agent_id][key], dtype=np.float32)

        return obs
    
class FeaturesObservationRL(Observation):
    def __init__(self, env, features: List[str], ego_idx: int = 0, large_num: float = 1e30):
        super().__init__(env)
        self.features = features
        self.large_num = large_num
        self.ego_idx = ego_idx
        
    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )

        ego_dict = {
            'scan': gym.spaces.Box(
                low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32
            ),
            'pose': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(3,), dtype=np.float32
            ),
            'vel': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(3,), dtype=np.float32
            ),
            'heading': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(2,), dtype=np.float32
            )
        }
        return gym.spaces.Dict(ego_dict)
    
    def observe(self):
        scan = self.env.unwrapped.sim.agent_scans[self.ego_idx]
        agent = self.env.unwrapped.sim.agents[self.ego_idx]
        std_state = agent.standard_state
        x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
        delta = std_state["delta"]
        beta = std_state["slip"]
        vx = std_state["v_x"]
        vy = std_state["v_y"]
        angvel = std_state["yaw_rate"]
        
        # create agent's observation dict
        return {
            "scan": scan.astype(np.float32),
            "pose": np.array((x,y,theta), dtype=np.float32),
            "vel": np.array((vx,vy,angvel), dtype=np.float32),
            "heading": np.array((delta, beta), dtype=np.float32)
        }

class VectorObservation(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        num_agents = len(self.env.unwrapped.agent_ids)
        assert num_agents == 1, "Vector observation only supports single agent"

        obs_size_dict = {
            "scan": scan_size,
            "pose_x": 1,
            "pose_y": 1,
            "pose_theta": 1,
            "linear_vel_x": 1,
            "linear_vel_y": 1,
            "ang_vel_z": 1,
            "delta": 1,
            "beta": 1,
            "collision": 1,
            "lap_time": 1,
            "lap_count": 1,
        }

        complete_space_size = sum([obs_size_dict[k] for k in self.features])

        obs_space = gym.spaces.Box(
            low=-large_num,
            high=large_num,
            shape=(complete_space_size,),
            dtype=float,
        )
        return obs_space

    def observe(self):
        scan = self.env.unwrapped.sim.agent_scans[0]
        agent = self.env.unwrapped.sim.agents[0]
        lap_time = self.env.unwrapped.lap_times[0]
        lap_count = self.env.unwrapped.lap_counts[0]

        std_state = agent.standard_state

        x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
        delta = std_state["delta"]
        beta = std_state["slip"]
        vx = std_state["v_x"]
        vy = std_state["v_y"]
        angvel = std_state["yaw_rate"]

        # create agent's observation dict
        agent_obs = {
            "scan": scan,
            "pose_x": x,
            "pose_y": y,
            "pose_theta": theta,
            "linear_vel_x": vx,
            "linear_vel_y": vy,
            "ang_vel_z": angvel,
            "delta": delta,
            "beta": beta,
            "collision": int(agent.in_collision),
            "lap_time": lap_time,
            "lap_count": lap_count,
        }

        # add agent's observation to multi-agent observation
        vec_obs = []
        for k in self.features:
            vec_obs.extend(list(agent_obs[k]))

        return np.array(vec_obs)
    
class FeaturesObservationMARL(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features
        self.large_num = 1e30
        self.num_agents = env.unwrapped.num_agents

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )

        space = {
            'scan': gym.spaces.Box(
                low=0.0, high=scan_range, shape=(self.num_agents, scan_size), dtype=np.float32
            ),
            'pose': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(self.num_agents, 3), dtype=np.float32
            ),
            'vel': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(self.num_agents, 3), dtype=np.float32
            ),
            'heading': gym.spaces.Box(
                low=-self.large_num, high=self.large_num, shape=(self.num_agents, 2), dtype=np.float32
            )
        }
        return gym.spaces.Dict(space)
    
    def observe(self):
        scans = self.env.unwrapped.sim.agent_scans
        poses = np.zeros((self.num_agents, 3), dtype=np.float32)
        vels = np.zeros((self.num_agents, 3), dtype=np.float32)
        headings = np.zeros((self.num_agents, 2), dtype=np.float32)
        agents = self.env.unwrapped.sim.agents
        for i in range(self.num_agents):
            agent = agents[i]
            std_state = agent.standard_state
            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]
            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            poses[i, :] = np.array((x,y,theta), dtype=np.float32)
            vels[i, :] = np.array((vx,vy,angvel), dtype=np.float32)
            headings[i, :] = np.array((delta, beta), dtype=np.float32)

        return {
            "scan": scans.astype(np.float32),
            "pose": poses,
            "vel": vels,
            "heading": headings
        }
    
class LIDARConvObservation(Observation):
    def __init__(self, env, features: List[str]):
        """Isolate LIDAR scans to be used with Conv1D networks"""
        super().__init__(env)
        self.features = features
        self.large_num = 1e30
        self.num_agents = env.unwrapped.num_agents

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )

        space = {
            'scan': gym.spaces.Box(
                low=0.0, high=scan_range, shape=(self.num_agents, scan_size), dtype=np.float32
            ),
            'odometry': gym.spaces.Box( # contains all of pose, velocity, and heading
                low=-self.large_num, high=self.large_num, shape=(self.num_agents, 8), dtype=np.float32
            )
        }

        return gym.spaces.Dict(space)
    
    def observe(self):
        scans = self.env.unwrapped.sim.agent_scans
        odometry = np.zeros((self.num_agents, 8), dtype=np.float32)
        agents = self.env.unwrapped.sim.agents
        for i in range(self.num_agents):
            agent = agents[i]
            std_state = agent.standard_state
            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]
            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            odometry[i, :] = np.array((x,y,theta,vx,vy,angvel,delta,beta), dtype=np.float32)

        return {
            "scan": scans.astype(np.float32),
            "odometry": odometry
        }
    
class MLPObservation(Observation):
    def __init__(self, env, features: List[str]):
        """
        Just observe a num_agents x num_features vector

        Should definitely be used with VecNormalize since the observation space is unbounded
        
        """
        super().__init__(env)
        self.features = features
        self.large_num = 1e30
        self.num_agents = env.unwrapped.num_agents

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        return gym.spaces.Box(
            low = -self.large_num, high = self.large_num, shape=(self.num_agents, scan_size + 8), dtype=np.float32
        )
    
    def observe(self):
        scans = self.env.unwrapped.sim.agent_scans
        odometry = np.zeros((self.num_agents, 8), dtype=np.float32)
        agents = self.env.unwrapped.sim.agents
        for i in range(self.num_agents):
            agent = agents[i]
            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]

            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            odometry[i, :] = np.array((x,y,theta,vx,vy,angvel,delta,beta), dtype=np.float32)

        return np.concatenate([scans.astype(np.float32), odometry], axis=1)


def observation_factory(env, type: str | None, **kwargs) -> Observation:
    type = type or "original"

    if type == "original":
        return OriginalObservation(env)
    elif type == "features":
        return FeaturesObservation(env, **kwargs)
    elif type == "kinematic_state":
        features = ["pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"]
        return FeaturesObservation(env, features=features)
    elif type == "dynamic_state":
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    elif type == "frenet_dynamic_state":
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "linear_vel_y",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    elif type == "rl":
        features = [
            "scan",
        ]
        return VectorObservation(env, features=features)
    elif type == "frenet_rl":
        features = [
            "scan",
            "pose_x",
            "pose_y",
            "pose_theta",
            "linear_vel_x",
            "linear_vel_y",
            "ang_vel_z",
            "delta",
            "beta"
        ]
        return FeaturesObservationRL(env, features=features)
    elif type == "frenet_marl":
        features = [
            "scan",
            "pose_x",
            "pose_y",
            "pose_theta",
            "linear_vel_x",
            "linear_vel_y",
            "ang_vel_z",
            "delta",
            "beta"
        ]
        return FeaturesObservationMARL(env, features=features)
    elif type == "lidar_conv":
        features = [
            "scan",
            "pose_x",
            "pose_y",
            "pose_theta",
            "linear_vel_x",
            "linear_vel_y",
            "ang_vel_z",
            "delta",
            "beta"
        ]
        return LIDARConvObservation(env, features=features)
    elif type == "mlp":
        features = [
            "scan",
            "pose_x",
            "pose_y",
            "pose_theta",
            "linear_vel_x",
            "linear_vel_y",
            "ang_vel_z",
            "delta",
            "beta"
        ]
        return MLPObservation(env, features=features)
    else:
        raise ValueError(f"Invalid observation type {type}.")
