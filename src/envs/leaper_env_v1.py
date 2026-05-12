import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src/models", "leaper_v1.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class LeaperEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    FOOT_NAMES = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
    FOOT_PHASE_OFFSETS = np.array([0.0, np.pi, np.pi, 0.0])

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 25,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.5,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        smoothness_weight: float = 0.02,
        angular_vel_weight: float = 0.05,
        posture_weight: float = 0.001,
        z_vel_weight: float = 0.02,
        gait_cost_weight: float = 0.1,
        gait_bonus_weight: float = 1.5,
        gait_frequency: float = 1.2,
        gait_duty: float = 0.6,
        target_velocity: float = 0.6,
        velocity_sigma: float = 0.4,
        foot_stance_threshold: float = 0.08,
        # ---
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        base_reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file, frame_skip, default_camera_config, forward_reward_weight,
            ctrl_cost_weight, contact_cost_weight, healthy_reward, smoothness_weight,
            angular_vel_weight, posture_weight, z_vel_weight, gait_cost_weight,
            gait_bonus_weight, gait_frequency, gait_duty, target_velocity, velocity_sigma,
            foot_stance_threshold, main_body, terminate_when_unhealthy,
            healthy_z_range, contact_force_range, reset_noise_scale, base_reset_noise_scale,
            exclude_current_positions_from_observation, include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._smoothness_weight = smoothness_weight
        self._angular_vel_weight = angular_vel_weight
        self._posture_weight = posture_weight
        self._z_vel_weight = z_vel_weight
        self._gait_cost_weight = gait_cost_weight
        self._gait_bonus_weight = gait_bonus_weight
        self._gait_frequency = gait_frequency
        self._gait_duty = gait_duty
        self._target_velocity = target_velocity
        self._velocity_sigma = velocity_sigma
        self._foot_stance_threshold = foot_stance_threshold

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._main_body = main_body
        self._reset_noise_scale = reset_noise_scale
        self._base_reset_noise_scale = base_reset_noise_scale

        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        xml_path = os.path.abspath(xml_file) if xml_file else os.path.abspath(_XML_PATH)

        MujocoEnv.__init__(
            self, xml_path, frame_skip, observation_space=None,
            default_camera_config=default_camera_config, **kwargs,
        )

        import mujoco
        numeric_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        if numeric_id >= 0:
            adr = self.model.numeric_adr[numeric_id]
            size = self.model.numeric_size[numeric_id]
            custom_qpos = self.model.numeric_data[adr:adr + size].copy()
            custom_qpos[7:] = np.deg2rad(custom_qpos[7:])
            self.init_qpos[:len(custom_qpos)] = custom_qpos

        self.default_joints = self.init_qpos[7:].copy()
        self.prev_action = np.zeros(self.model.nu)
        self._phase = 0.0
        self._phase_dt = 2.0 * np.pi * self._gait_frequency * self.dt

        self._foot_body_ids = []
        for name in self.FOOT_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert bid >= 0, f"Body '{name}' not found in XML"
            self._foot_body_ids.append(bid)

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        obs_size += 2  

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return np.isfinite(state).all() and min_z <= state[2] <= max_z

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def contact_forces(self):
        return np.clip(self.data.cfrc_ext, self._contact_force_range[0], self._contact_force_range[1])

    @property
    def contact_cost(self):
        return self._contact_cost_weight * np.sum(np.square(self.contact_forces))

    def _gait_error(self):
        error_count = 0.0
        duty_cutoff = self._gait_duty * 2.0 * np.pi
        for i, bid in enumerate(self._foot_body_ids):
            foot_z = self.data.body(bid).xpos[2]
            phi = (self._phase + self.FOOT_PHASE_OFFSETS[i]) % (2.0 * np.pi)
            target_stance = phi < duty_cutoff
            actual_stance = foot_z < self._foot_stance_threshold
            if target_stance != actual_stance:
                error_count += 1.0
        return error_count

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        x_velocity, y_velocity = (xy_position_after - xy_position_before) / self.dt

        self._phase = (self._phase + self._phase_dt) % (2.0 * np.pi)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        gait_errors = self._gait_error()
        compliance = 1.0 - (gait_errors / len(self.FOOT_NAMES))

        v_err = abs(x_velocity - self._target_velocity) / self._velocity_sigma
        forward_reward = self._forward_reward_weight * max(0.0, 1.0 - v_err)
        healthy_reward = self.healthy_reward
        gait_bonus = self._gait_bonus_weight * compliance

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        smoothness_cost = self._smoothness_weight * np.sum(np.square(action - self.prev_action))
        self.prev_action = np.copy(action)
        angular_vel_cost = self._angular_vel_weight * np.sum(np.square(self.data.qvel[3:5]))
        z_vel_cost = self._z_vel_weight * np.square(self.data.qvel[2])
        posture_cost = self._posture_weight * np.sum(np.square(self.data.qpos[7:] - self.default_joints))
        gait_cost = self._gait_cost_weight * gait_errors

        costs = (ctrl_cost + contact_cost + smoothness_cost + angular_vel_cost +
                 z_vel_cost + posture_cost + gait_cost)

        reward = forward_reward + healthy_reward + gait_bonus - costs

        return reward, {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_gait_bonus": gait_bonus,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_smoothness": -smoothness_cost,
            "reward_angular_vel": -angular_vel_cost,
            "reward_z_vel": -z_vel_cost,
            "reward_posture": -posture_cost,
            "reward_gait_penalty": -gait_cost,
            "gait_compliance": compliance,
        }

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        phase_obs = np.array([np.sin(self._phase), np.cos(self._phase)])

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force, phase_obs))
        else:
            return np.concatenate((position, velocity, phase_obs))

    def reset_model(self):
        js, bs = self._reset_noise_scale, self._base_reset_noise_scale
        qpos = self.init_qpos.copy()
        qpos[:7] += self.np_random.uniform(low=-bs, high=bs, size=7)
        qpos[7:] += self.np_random.uniform(low=-js, high=js, size=self.model.nq - 7)
        qvel = self.init_qvel.copy()
        qvel[:6] += bs * self.np_random.standard_normal(6)
        qvel[6:] += js * self.np_random.standard_normal(self.model.nv - 6)
        self.set_state(qpos, qvel)
        self.prev_action = np.zeros(self.model.nu)
        self._phase = 0.0
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }