from __future__ import annotations

import argparse
import os
from typing import Dict

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio
import numpy as np
import gymnasium as gym

import src.envs


_D = np.pi / 180.0

LEGS = ['l1', 'l2', 'l3', 'r1', 'r2', 'r3']

_ANKLE_FRONT_DEG = +30.0
_ANKLE_MID_DEG   = +30.0
_ANKLE_REAR_DEG  = +30.0

def _stance_pose(leg: str) -> Dict[str, float]:
    if leg in ('l1', 'r1'):
        a = _ANKLE_FRONT_DEG
    elif leg in ('l2', 'r2'):
        a = _ANKLE_MID_DEG
    else:
        a = _ANKLE_REAR_DEG
    return {'hip': 0.0, 'ankle': a * _D}

STANCE: Dict[str, Dict[str, float]] = {leg: _stance_pose(leg) for leg in LEGS}

HIP_SIGN = {'l1': -1, 'l2': -1, 'l3': -1,
            'r1': +1, 'r2': +1, 'r3': +1}

PHASE_OFFSET = {
    'l1': np.pi, 'r2': np.pi, 'l3': np.pi,
    'r1': 0.0,   'l2': 0.0,   'r3': 0.0,
}

ACT_ORDER = [(leg, j) for leg in LEGS for j in ['hip', 'ankle']]
J2IDX = {f"{leg}_{j}": i for i, (leg, j) in enumerate(ACT_ORDER)}


def _env_id(env_name: str) -> str:
    return 'Bastion-v1'


def _stance_qpos_vector() -> np.ndarray:
    q = np.zeros(12)
    for leg in LEGS:
        s = STANCE[leg]
        q[J2IDX[f'{leg}_hip']]   = s['hip']
        q[J2IDX[f'{leg}_ankle']] = s['ankle']
    return q


def reset_env_to_stance(env, seed: int = None) -> np.ndarray:
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    u = env.unwrapped

    qpos = np.zeros_like(u.init_qpos)
    qpos[0:3]  = [0.0, 0.0, 0.50]
    qpos[3:7]  = [1.0, 0.0, 0.0, 0.0]
    qpos[7:19] = _stance_qpos_vector()

    qvel = np.zeros_like(u.init_qvel)
    u.set_state(qpos, qvel)
    return u._get_obs()


class ScriptedCPGControllerBastion:

    PRESETS = {
        'slow':   dict(freq=1.2, duty=0.65, hip_amp_deg=22,
                       ankle_lift_front_deg=35,
                       ankle_lift_middle_deg=35,
                       ankle_lift_rear_deg=35,
                       ankle_push_deg=15),
        'medium': dict(freq=1.5, duty=0.60, hip_amp_deg=26,
                       ankle_lift_front_deg=45,
                       ankle_lift_middle_deg=45,
                       ankle_lift_rear_deg=45,
                       ankle_push_deg=20),
        'fast':   dict(freq=1.8, duty=0.55, hip_amp_deg=30,
                       ankle_lift_front_deg=55,
                       ankle_lift_middle_deg=55,
                       ankle_lift_rear_deg=55,
                       ankle_push_deg=25),
    }

    _PHASE_INIT = 0.0

    def __init__(self, env, robot: str = 'bastion',
                 preset: str = 'medium',
                 freq: float = None, duty: float = None,
                 hip_amp_deg: float = None,
                 ankle_lift_front_deg: float = None,
                 ankle_lift_middle_deg: float = None,
                 ankle_lift_rear_deg: float = None,
                 ankle_push_deg: float = None,
                 kp: float = 0.8, kd: float = 0.15,
                 kp_ramp_steps: int = 20, kp_ramp_start_frac: float = 0.3,
                 settle_steps: int = 15):
        if robot.lower() != 'bastion':
            raise ValueError(f"BastionController tuned for Bastion only. Got robot='{robot}'.")

        self.env = env.unwrapped
        self.dt = self.env.dt

        p = dict(self.PRESETS[preset])
        for k, v in (('freq', freq), ('duty', duty),
                     ('hip_amp_deg', hip_amp_deg),
                     ('ankle_lift_front_deg', ankle_lift_front_deg),
                     ('ankle_lift_middle_deg', ankle_lift_middle_deg),
                     ('ankle_lift_rear_deg', ankle_lift_rear_deg),
                     ('ankle_push_deg', ankle_push_deg)):
            if v is not None: p[k] = v

        self.freq = p['freq']
        self.duty = p['duty']
        self.hip_amp              = p['hip_amp_deg']              * _D
        self.ankle_lift_front     = p['ankle_lift_front_deg']     * _D
        self.ankle_lift_mid       = p['ankle_lift_middle_deg']    * _D
        self.ankle_lift_rear      = p['ankle_lift_rear_deg']      * _D
        self.ankle_push           = p['ankle_push_deg']           * _D

        self.kp = kp
        self.kd = kd
        self.kp_ramp_steps = kp_ramp_steps
        self.kp_ramp_start_frac = kp_ramp_start_frac
        self.settle_steps = settle_steps

        self.phase = self._PHASE_INIT
        self.t = 0
        self.phase_dt = 2.0 * np.pi * self.freq * self.dt

    def reset(self) -> None:
        self.phase = self._PHASE_INIT
        self.t = 0

    def _leg_targets(self, leg: str, phi: float):
        u = (phi % (2 * np.pi)) / (2 * np.pi)
        front  = leg in ('l1', 'r1')
        middle = leg in ('l2', 'r2')
        stance_pose = STANCE[leg]

        if u < self.duty:
            sf = u / self.duty
            hip_factor = 1.0 - 2.0 * sf
            bell = np.sin(np.pi * sf)
            ankle_tgt = stance_pose['ankle'] - self.ankle_push * bell
        else:
            wf = (u - self.duty) / (1.0 - self.duty)
            hip_factor  = 2.0 * wf - 1.0
            swing_bell  = np.sin(np.pi * wf)
            if front:
                lift = self.ankle_lift_front
            elif middle:
                lift = self.ankle_lift_mid
            else:
                lift = self.ankle_lift_rear
            ankle_tgt = stance_pose['ankle'] + lift * swing_bell

        hip_tgt = HIP_SIGN[leg] * self.hip_amp * hip_factor
        return hip_tgt, ankle_tgt

    def desired_qpos(self, phase: float = None) -> np.ndarray:
        if phase is None:
            phase = self.phase
        if self.t < self.settle_steps:
            return _stance_qpos_vector()
        q = np.zeros(12)
        for leg in LEGS:
            phi = phase + PHASE_OFFSET[leg]
            h, a = self._leg_targets(leg, phi)
            q[J2IDX[f'{leg}_hip']]   = h
            q[J2IDX[f'{leg}_ankle']] = a
        return q

    def _effective_kp(self) -> float:
        n = self.kp_ramp_steps
        if n <= 0 or self.t >= n:
            return self.kp
        f0 = self.kp_ramp_start_frac
        return self.kp * (f0 + (1.0 - f0) * (self.t + 1) / n)

    def act(self) -> np.ndarray:
        q_des = self.desired_qpos()
        q  = self.env.data.qpos[7:19]
        qd = self.env.data.qvel[6:18]
        u = self._effective_kp() * (q_des - q) - self.kd * qd
        return np.clip(u, -1.0, 1.0)

    def step_phase(self) -> None:
        self.t += 1
        if self.t <= self.settle_steps:
            self.phase = self._PHASE_INIT
            return
        self.phase = (self.phase + self.phase_dt) % (2.0 * np.pi)


def collect_episode(env, ctrl: ScriptedCPGControllerBastion,
                    max_ep_len: int, seed: int):
    obs = reset_env_to_stance(env, seed=seed)
    ctrl.reset()
    ep = {'observations': [obs.copy()], 'actions': [], 'rewards': [],
          'terminations': [], 'truncations': []}
    for t in range(max_ep_len):
        action = ctrl.act()
        obs_next, reward, term, trunc, _ = env.step(action)
        ctrl.step_phase()
        ep['actions'].append(action.copy())
        ep['rewards'].append(float(reward))
        ep['terminations'].append(bool(term))
        ep['truncations'].append(bool(trunc or (t + 1) >= max_ep_len))
        ep['observations'].append(obs_next.copy())
        if term or trunc:
            break
    return ep


def save_demos_hdf5(env_name: str, out_path: str, n_episodes: int = 20,
                    max_ep_len: int = 1000, seed: int = 0,
                    preset: str = 'medium') -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    env = gym.make(_env_id(env_name))
    ctrl = ScriptedCPGControllerBastion(env, robot=env_name, preset=preset)
    returns = []
    with h5py.File(out_path, 'w') as f:
        for i in range(n_episodes):
            ep = collect_episode(env, ctrl, max_ep_len, seed=seed + i)
            g = f.create_group(f"episode_{i}")
            for k, v in ep.items():
                g.create_dataset(k, data=np.array(v))
            returns.append(float(np.sum(ep['rewards'])))
        f.attrs['env'] = env_name
        f.attrs['n_episodes'] = n_episodes
        f.attrs['preset'] = preset
    env.close()
    print(f"[cpg-bastion] {env_name} ({preset}): {n_episodes} eps -> {out_path} | "
          f"return mean={np.mean(returns):.1f} std={np.std(returns):.1f} "
          f"min={np.min(returns):.1f} max={np.max(returns):.1f}")


def record_video(env_name: str, out_path: str, max_ep_len: int = 1000,
                 seed: int = 0, preset: str = 'medium') -> None:
    env = gym.make(_env_id(env_name), render_mode='rgb_array')
    ctrl = ScriptedCPGControllerBastion(env, robot=env_name, preset=preset)
    reset_env_to_stance(env, seed=seed)
    ctrl.reset()
    frames = []
    total_r = 0.0
    for t in range(max_ep_len):
        frames.append(env.render())
        action = ctrl.act()
        _, r, term, trunc, _ = env.step(action)
        ctrl.step_phase()
        total_r += r
        if term or trunc:
            break
    env.close()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fps = int(round(1.0 / env.unwrapped.dt))
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[cpg-bastion] {env_name} ({preset}): {len(frames)} frames, "
          f"return={total_r:.1f} -> {out_path}")


def seed_replay_buffer(buffer, env_name: str, n_transitions: int,
                       max_ep_len: int = 1000, seed: int = 0,
                       preset: str = 'medium') -> int:
    env = gym.make(_env_id(env_name))
    ctrl = ScriptedCPGControllerBastion(env, robot=env_name, preset=preset)
    pushed = 0
    ep = 0
    while pushed < n_transitions:
        obs = reset_env_to_stance(env, seed=seed + ep)
        ctrl.reset()
        for t in range(max_ep_len):
            action = ctrl.act()
            obs_next, reward, term, trunc, _ = env.step(action)
            ctrl.step_phase()
            done_bootstrap = term and (t + 1) < max_ep_len
            buffer.store(obs, action, reward, obs_next, float(done_bootstrap))
            pushed += 1
            obs = obs_next
            if term or trunc or pushed >= n_transitions:
                break
        ep += 1
    env.close()
    print(f"[cpg-bastion] seeded buffer with {pushed} scripted transitions "
          f"from {env_name} ({preset})")
    return pushed


def imitation_reward(ctrl: ScriptedCPGControllerBastion,
                     w_q: float = 1.0, w_qdot: float = 0.0) -> float:
    q_des = ctrl.desired_qpos()
    q = ctrl.env.data.qpos[7:19]
    r_q = -w_q * float(np.mean((q - q_des) ** 2))
    if w_qdot > 0.0:
        qdot = ctrl.env.data.qvel[6:18]
        r_q -= w_qdot * float(np.mean(qdot ** 2))
    return r_q


def _parse():
    p = argparse.ArgumentParser(
        description="Alternating-tripod CPG for the ARC-RL Bastion"
    )
    p.add_argument('--env', type=str, default='bastion', choices=['bastion'])
    p.add_argument('--preset', type=str, default='medium',
                   choices=['slow', 'medium', 'fast'])
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max-ep-len', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=None, help='HDF5 path for demos')
    p.add_argument('--video', type=str, default=None, help='MP4 path')
    return p.parse_args()


def main():
    args = _parse()
    if args.video:
        record_video(args.env, args.video, max_ep_len=args.max_ep_len,
                     seed=args.seed, preset=args.preset)
    if args.out:
        save_demos_hdf5(args.env, args.out, n_episodes=args.episodes,
                        max_ep_len=args.max_ep_len, seed=args.seed,
                        preset=args.preset)
    if not args.video and not args.out:
        env = gym.make(_env_id(args.env))
        ctrl = ScriptedCPGControllerBastion(env, robot=args.env, preset=args.preset)
        ep = collect_episode(env, ctrl, args.max_ep_len, seed=args.seed)
        env.close()
        print(f"[cpg-bastion] {args.env} ({args.preset}): "
              f"{len(ep['actions'])} steps, return={np.sum(ep['rewards']):.2f}")


if __name__ == '__main__':
    main()