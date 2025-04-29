#!/usr/bin/env python3
"""An MPPI based planner."""
import jax
import jax.numpy as jnp
import os, sys
sys.path.append("../")
from functools import partial
import numpy as np
from jax import debug
from jax import lax

class MPPI():
    """An MPPI based planner."""
    def __init__(self, config, env, jrng, 
                 temperature=0.005, damping=0.001, track=None):
        self.config = config
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.temperature = temperature
        self.damping = damping
        self.a_std = jnp.array(config.control_sample_std)
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (config.adaptive_covariance and self.n_iterations > 1) or self.a_cov_shift
        self.a_shape = config.control_dim
        self.env = env
        self.jrng = jrng
        self.init_state(self.env, self.a_shape)
        self.accum_matrix = jax.device_put(jnp.triu(jnp.ones((self.n_steps, self.n_steps))))
        self.track = track
        self.a_cov_reset = (self.a_std**2) * jnp.eye(self.a_shape)

    def init_state(self, env, a_shape):
        # uses random as a hack to support vmap
        dim_a = jnp.prod(a_shape)
        self.env = env
        self.a_opt = 0.0 * jax.random.uniform(self.jrng.new_key(), shape=(self.n_steps, dim_a))
        # a_cov: [n_steps, dim_a, dim_a]
        if self.a_cov_shift:
            self.a_cov = (self.a_std**2) * jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = self.a_cov

    @partial(jax.jit, static_argnums=(0,))
    def mppi_loop(self, a_opt, a_cov, env_state, reference_traj, obs_array, key):
        def body_fn(carry, _):
            a_opt, a_cov, key = carry
            key, subkey = jax.random.split(key)
            a_opt, a_cov, states, traj = self.iteration_step(a_opt, a_cov, subkey, env_state, reference_traj, obs_array)
            return (a_opt, a_cov, key), (states, traj)

        (a_opt, a_cov, _), (all_states, all_trajs) = lax.scan(
            body_fn,
            (a_opt, a_cov, key),
            xs=None,
            length=self.n_iterations
        )
        return a_opt, a_cov, all_states[-1], all_trajs[-1]
    
    def update(self, env_state, reference_traj, obs_array):
        self.a_opt, self.a_cov = self.shift_prev_opt(self.a_opt, self.a_cov)
        self.a_opt, self.a_cov, self.states, self.traj_opt = self.mppi_loop(
            self.a_opt, self.a_cov, env_state, reference_traj, obs_array, self.jrng.new_key()
        )
    
        '''
        if self.track is not None and self.config.state_predictor in self.config.cartesian_models:
            self.states = self.convert_cartesian_to_frenet_jax(self.states)
            self.traj_opt = self.convert_cartesian_to_frenet_jax(self.traj_opt)
        '''
        self.sampled_states = self.states

    @partial(jax.jit, static_argnums=(0,))
    def shift_prev_opt(self, a_opt, a_cov):
        a_opt = a_opt.at[:-1].set(a_opt[1:])
        a_opt = a_opt.at[-1].set(jnp.zeros((self.a_shape,)))
        if self.a_cov_shift:
            a_cov = a_cov.at[:-1].set(a_cov[1:])
            a_cov = a_cov.at[-1].set(self.a_cov_reset)
        else:
            a_cov = self.a_cov_init
        return a_opt, a_cov

    @staticmethod
    @partial(jax.jit, static_argnames=("collision_radius",))
    def check_collision_batch(states, obstacle_points, collision_radius=0.3):
        def check_state_timestep(state_t):
            dx = obstacle_points[:, 0] - state_t[0]
            dy = obstacle_points[:, 1] - state_t[1]
            dist_sq = dx**2 + dy**2
            return jnp.any(dist_sq < collision_radius**2)
        return jax.vmap(check_state_timestep)(states)
    
    @partial(jax.jit, static_argnums=(0,))
    def mask_after_collision(self, collision_flags):
        collision_cumsum = jnp.cumsum(collision_flags.astype(jnp.int32))
        return collision_cumsum > 0
    
    @partial(jax.jit, static_argnums=(0,))
    def iteration_step(self, a_opt, a_cov, rng_da, env_state, reference_traj, obs_array):
        rng_da, rng_da_split1, rng_da_split2 = jax.random.split(rng_da, 3)
        da, actions, states = self._jit_rollout_block(a_opt, rng_da, rng_da_split1, env_state)
        
        # states: [n_samples, T, state_dim]
        # obs_array: [max_obs, 2] (padded with invalid values, e.g. self.config.invalid_pos)
        collision_flags = jax.vmap(self.check_collision_batch, in_axes=(0, None))(states, obs_array)
        collision_mask = jax.vmap(self.mask_after_collision)(collision_flags)
        if self.config.state_predictor in self.config.cartesian_models:
            reward = jax.vmap(self.env.reward_fn_xy, in_axes=(0, None))(states, reference_traj)
        else:
            reward = jax.vmap(self.env.reward_fn_sey, in_axes=(0, None))(states, reference_traj)
        reward = jnp.where(collision_mask, -100.0, reward)
        
        flat_states = states.reshape((-1, states.shape[-1]))
        frenet = self.track.vmap_cartesian_to_frenet_jax(flat_states[:, (0, 1, 4)])
        d_vals = frenet[:, 1]
        off_track = jnp.abs(d_vals) > 0.5
        off_track = off_track.reshape(states.shape[0], states.shape[1])
        off_track_cumsum = jnp.cumsum(off_track.astype(jnp.int32), axis=1)
        off_track_mask = off_track_cumsum > 0
        reward = jnp.where(off_track_mask, -100.0, reward)
        
        a_opt, a_cov = self._jit_optimization_block(a_opt, a_cov, da, reward)
        if self.config.render:
            traj_opt = self.rollout(a_opt, env_state, rng_da_split2)
        else:
            traj_opt = states[0]
        return a_opt, a_cov, states, traj_opt
    
    @partial(jax.jit, static_argnums=(0,))
    def _jit_rollout_block(self, a_opt, rng_da, rng_split_key, env_state):
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * self.a_std - a_opt,
            jnp.ones_like(a_opt) * self.a_std - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape)
        )
        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        env_states = jnp.tile(env_state[None, :], (actions.shape[0], 1))
        states = self.rollout_batch(actions, env_states, rng_split_key)
        return da, actions, states
    
    @partial(jax.jit, static_argnums=(0,))
    def _jit_optimization_block(self, a_opt, a_cov, da, reward):
        R = jax.vmap(self.returns)(reward)  # [n_samples, n_steps]
        w = jax.vmap(self.weights, 1, 1)(R)    # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)
        if self.adaptive_covariance:
            a_cov = da[..., :, None] * da[..., None, :]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(a_cov, 0, w)
            a_cov = a_cov + jnp.eye(self.a_shape) * 1e-5
        return a_opt, a_cov

    @partial(jax.jit, static_argnums=(0,))
    def returns(self, r):
        return jnp.dot(self.accum_matrix, r)

    @partial(jax.jit, static_argnums=(0,))
    def weights(self, R):
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        w = jnp.exp(R_stdzd / self.temperature)
        w = w / jnp.sum(w)
        return w

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, actions, env_state, rng_key):
        def rollout_step(carry, action):
            env_state = carry
            action = jnp.reshape(action, self.env.a_shape)
            env_state, _, _ = self.env.step(env_state, action, rng_key)
            return env_state, env_state
        _, states = lax.scan(rollout_step, env_state, actions)
        return states
        
    @partial(jax.jit, static_argnums=(0,))
    def rollout_batch(self, actions_batch, env_states, rng_key):
        return jax.vmap(self.rollout, in_axes=(0, 0, None))(actions_batch, env_states, rng_key)
    
    @partial(jax.jit, static_argnums=(0,))
    def convert_cartesian_to_frenet_jax(self, states):
        states_shape = (*states.shape[:-1], 7)
        states = states.reshape(-1, states.shape[-1])
        converted_states = self.track.vmap_cartesian_to_frenet_jax(states[:, (0, 1, 4)])
        states_frenet = jnp.concatenate([
            converted_states[:, :2], 
            states[:, 2:4] * jnp.cos(states[:, 6:7]),
            converted_states[:, 2:3],
            states[:, 2:4] * jnp.sin(states[:, 6:7])
        ], axis=-1)
        return states_frenet.reshape(states_shape)