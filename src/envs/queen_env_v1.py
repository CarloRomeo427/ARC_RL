import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src/models", "queen_v1.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 6.0,
}

class QueenEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    FOOT_NAMES = ["l1_foot", "r1_foot", "l2_foot", "r2_foot", "l3_foot", "r3_foot"]
    FOOT_PHASE_OFFSETS = np.array([0.0, np.pi, np.pi, 0.0, 0.0, np.pi])

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 25,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.5,
        ctrl_cost_weight: float = 0.05,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        smoothness_weight: float = 0.02,
        angular_vel_weight: float = 0.05,
        posture_weight: float = 0.001, 
        z_vel_weight: float = 0.02,
        gait_cost_weight: float = 0.25, 
        orientation_weight: float = 0.5,
        gait_frequency: float = 1.5,
        target_velocity: float = 1.5,  # m/s — speed cap for hexapod
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.25, 1.2),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, default_camera_config, forward_reward_weight, 
            ctrl_cost_weight, contact_cost_weight, healthy_reward, smoothness_weight, 
            angular_vel_weight, posture_weight, z_vel_weight, gait_cost_weight, 
            orientation_weight, gait_frequency, target_velocity, main_body, terminate_when_unhealthy, 
            healthy_z_range, contact_force_range, reset_noise_scale, 
            exclude_current_positions_from_observation, include_cfrc_ext_in_observation, **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._smoothness_weight = smoothness_weight
        self._angular_vel_weight = angular_vel_weight
        self._posture_weight = posture_weight
        self._z_vel_weight = z_vel_weight
        self._gait_cost_weight = gait_cost_weight
        self._orientation_weight = orientation_weight
        self._gait_frequency = gait_frequency
        self._target_velocity = target_velocity
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._main_body = main_body
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        xml_path = os.path.abspath(xml_file) if xml_file else os.path.abspath(_XML_PATH)

        MujocoEnv.__init__(
            self, xml_path, frame_skip, observation_space=None, 
            default_camera_config=default_camera_config, **kwargs
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
            "render_fps": int(np.round(1.0 / self.dt))
        }

        obs_size = (self.data.qpos.size + self.data.qvel.size 
                    - 2 * exclude_current_positions_from_observation 
                    + self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation + 2)
        
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
        for i, bid in enumerate(self._foot_body_ids):
            foot_z = self.data.body(bid).xpos[2]
            target_stance = np.sin(self._phase + self.FOOT_PHASE_OFFSETS[i]) <= 0
            actual_stance = foot_z < 0.12  
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
        # 1. Calculate Gait Compliance
        gait_errors = self._gait_error()
        compliance = 1.0 - (gait_errors / len(self.FOOT_NAMES))
        
        # 2. Dance Reward (Rewards form over speed)
        compliance_bonus = compliance * 2.0 
        
        # 3. Gate the Velocity Reward (capped to prevent rushing)
        capped_velocity = min(x_velocity, self._target_velocity)
        forward_reward = (capped_velocity * self._forward_reward_weight) * compliance
        healthy_reward = self.healthy_reward

        # 4. Softened Tripod Check
        legs_on_ground = sum(1 for bid in self._foot_body_ids if self.data.body(bid).xpos[2] < 0.12)
        tripod_penalty = 0.0
        if legs_on_ground < 3:
            tripod_penalty = 0.5 * (3 - legs_on_ground) 

        # Standard Costs
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        smoothness_cost = self._smoothness_weight * np.sum(np.square(action - self.prev_action))
        self.prev_action = np.copy(action)
        angular_vel_cost = self._angular_vel_weight * np.sum(np.square(self.data.qvel[3:5]))
        z_vel_cost = self._z_vel_weight * np.square(self.data.qvel[2])
        posture_cost = self._posture_weight * np.sum(np.square(self.data.qpos[7:] - self.default_joints))
        
        # Direct Gate & Orientation Costs
        gait_cost = self._gait_cost_weight * gait_errors
        # STRICT Pitch Gating: qpos[4:6] holds the [x, y] quaternions. y controls pitch.
        orientation_cost = self._orientation_weight * np.sum(np.square(self.data.qpos[4:6]))

        costs = (ctrl_cost + contact_cost + smoothness_cost + angular_vel_cost + 
                 z_vel_cost + posture_cost + gait_cost + orientation_cost + tripod_penalty)

        reward = forward_reward + healthy_reward + compliance_bonus - costs

        return reward, {
            "reward_forward": forward_reward, 
            "reward_survive": healthy_reward,
            "reward_compliance_bonus": compliance_bonus,
            "reward_ctrl": -ctrl_cost, 
            "reward_contact": -contact_cost,
            "reward_smoothness": -smoothness_cost, 
            "reward_angular_vel": -angular_vel_cost,
            "reward_z_vel": -z_vel_cost, 
            "reward_posture": -posture_cost,
            "reward_gait_penalty": -gait_cost, 
            "reward_orientation_penalty": -orientation_cost,
            "reward_tripod_penalty": -tripod_penalty,
            "gait_compliance": compliance
        }

    def _get_obs(self):
        position, velocity = self.data.qpos.flatten(), self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation: 
            position = position[2:]
        phase_obs = np.array([np.sin(self._phase), np.cos(self._phase)])
        
        if self._include_cfrc_ext_in_observation:
            return np.concatenate((position, velocity, self.contact_forces[1:].flatten(), phase_obs))
        return np.concatenate((position, velocity, phase_obs))

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_action, self._phase = np.zeros(self.model.nu), 0.0
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0], 
            "y_position": self.data.qpos[1], 
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2)
        }