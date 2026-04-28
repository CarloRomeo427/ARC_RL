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

STANCE: Dict[str, Dict[str, float]] = {
    'fl': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
    'fr': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
    'rl': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
    'rr': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
}

HIP_SIGN = {'fl': -1, 'fr': +1, 'rl': -1, 'rr': +1}
PHASE_OFFSET = {'fl': 0.0, 'rr': 0.0, 'fr': np.pi, 'rl': np.pi}

ACT_ORDER = [(leg, j) for leg in ['fl', 'fr', 'rl', 'rr']
             for j in ['hip', 'knee', 'ankle']]
J2IDX = {f"{leg}_{j}": i for i, (leg, j) in enumerate(ACT_ORDER)}


def _env_id(env_name: str) -> str:
    return 'Leaper-v1'


class ScriptedCPGControllerLeaper:

    PRESETS = {
        'slow':   dict(freq=1.0, duty=0.65, hip_amp_deg=30,
                       knee_flex_deg=-50, ankle_lift_deg=70,
                       front_push_deg=20),
        'medium': dict(freq=1.2, duty=0.60, hip_amp_deg=30,
                       knee_flex_deg=-55, ankle_lift_deg=75,
                       front_push_deg=25),
        'fast':   dict(freq=1.5, duty=0.60, hip_amp_deg=30,
                       knee_flex_deg=-60, ankle_lift_deg=80,
                       front_push_deg=30),
    }

    def __init__(self, env, robot: str = 'leaper',
                 preset: str = 'fast',
                 freq: float = None, duty: float = None,
                 hip_amp_deg: float = None,
                 knee_flex_deg: float = None,
                 ankle_lift_deg: float = None,
                 front_push_deg: float = None,
                 kp: float = 0.8, kd: float = 0.15,
                 kp_ramp_steps: int = 15, kp_ramp_start_frac: float = 0.2):
        if robot.lower() != 'leaper':
            raise ValueError(f"Controller tuned for Leaper only. Got robot='{robot}'.")
            
        self.env = env.unwrapped
        self.dt = self.env.dt

        p = dict(self.PRESETS[preset])
        for k, v in (('freq', freq), ('duty', duty),
                     ('hip_amp_deg', hip_amp_deg),
                     ('knee_flex_deg', knee_flex_deg),
                     ('ankle_lift_deg', ankle_lift_deg),
                     ('front_push_deg', front_push_deg)):
            if v is not None: p[k] = v

        self.freq = p['freq']
        self.duty = p['duty']
        self.hip_amp    = p['hip_amp_deg']    * _D
        self.knee_flex  = p['knee_flex_deg']  * _D
        self.ankle_mag  = p['ankle_lift_deg'] * _D
        self.front_push = p['front_push_deg'] * _D

        self.kp = kp
        self.kd = kd
        self.kp_ramp_steps = kp_ramp_steps
        self.kp_ramp_start_frac = kp_ramp_start_frac

        self.phase = 0.0
        self.t = 0
        self.phase_dt = 2.0 * np.pi * self.freq * self.dt

    def reset(self) -> None:
        self.phase = 0.0
        self.t = 0

    def _leg_targets(self, leg: str, phi: float):
        u = (phi % (2 * np.pi)) / (2 * np.pi)
        front = leg in ('fl', 'fr')
        ankle_sign = +1.0 if front else -1.0

        if u < self.duty:
            sf = u / self.duty
            hip_factor = 1.0 - 2.0 * sf
            knee_tgt   = STANCE[leg]['knee']
            if front:
                push_level = np.sin(np.pi * sf)
                ankle_tgt  = STANCE[leg]['ankle'] + self.front_push * push_level
            else:
                ankle_tgt  = STANCE[leg]['ankle']
        else:
            wf = (u - self.duty) / (1.0 - self.duty)
            hip_factor  = 2.0 * wf - 1.0
            swing_level = np.sin(np.pi * wf)
            knee_tgt    = STANCE[leg]['knee']  + self.knee_flex * swing_level
            ankle_tgt   = STANCE[leg]['ankle'] + ankle_sign * self.ankle_mag * swing_level

        hip_tgt = HIP_SIGN[leg] * self.hip_amp * hip_factor
        return hip_tgt, knee_tgt, ankle_tgt

    def desired_qpos(self, phase: float = None) -> np.ndarray:
        if phase is None:
            phase = self.phase
        q = np.zeros(12)
        for leg in ['fl', 'fr', 'rl', 'rr']:
            phi = phase + PHASE_OFFSET[leg]
            h, k, a = self._leg_targets(leg, phi)
            q[J2IDX[f'{leg}_hip']]   = h
            q[J2IDX[f'{leg}_knee']]  = k
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
        q = self.env.data.qpos[7:19]
        qd = self.env.data.qvel[6:18]
        u = self._effective_kp() * (q_des - q) - self.kd * qd
        return np.clip(u, -1.0, 1.0)

    def step_phase(self) -> None:
        self.t += 1
        self.phase = (self.phase + self.phase_dt) % (2.0 * np.pi)


def collect_episode(env, ctrl: ScriptedCPGControllerLeaper, max_ep_len: int, seed: int):
    obs, _ = env.reset(seed=seed)
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
                    preset: str = 'fast') -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    env = gym.make(_env_id(env_name))
    ctrl = ScriptedCPGControllerLeaper(env, robot=env_name, preset=preset)
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
    print(f"[cpg] {env_name} ({preset}): {n_episodes} eps -> {out_path} | "
          f"return mean={np.mean(returns):.1f} std={np.std(returns):.1f} "
          f"min={np.min(returns):.1f} max={np.max(returns):.1f}")

def record_video(env_name: str, out_path: str, max_ep_len: int = 1000,
                 seed: int = 0, preset: str = 'fast') -> None:
    env = gym.make(_env_id(env_name), render_mode='rgb_array')
    ctrl = ScriptedCPGControllerLeaper(env, robot=env_name, preset=preset)
    obs, _ = env.reset(seed=seed)
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
    print(f"[cpg] {env_name} ({preset}): {len(frames)} frames, "
          f"return={total_r:.1f} -> {out_path}")

def seed_replay_buffer(buffer, env_name: str, n_transitions: int,
                       max_ep_len: int = 1000, seed: int = 0,
                       preset: str = 'fast') -> int:
    env = gym.make(_env_id(env_name))
    ctrl = ScriptedCPGControllerLeaper(env, robot=env_name, preset=preset)
    pushed = 0
    ep = 0
    while pushed < n_transitions:
        obs, _ = env.reset(seed=seed + ep)
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
    print(f"[cpg] seeded buffer with {pushed} scripted transitions "
          f"from {env_name} ({preset})")
    return pushed

def imitation_reward(ctrl: ScriptedCPGControllerLeaper,
                     w_q: float = 1.0, w_qdot: float = 0.0) -> float:
    q_des = ctrl.desired_qpos()
    q = ctrl.env.data.qpos[7:19]
    r_q = -w_q * float(np.mean((q - q_des) ** 2))
    if w_qdot > 0.0:
        qdot = ctrl.env.data.qvel[6:18]
        r_q -= w_qdot * float(np.mean(qdot ** 2))
    return r_q

def _parse():
    p = argparse.ArgumentParser(description="Trot CPG w/ stance push for ARC-RL Leaper")
    p.add_argument('--env', type=str, default='leaper', choices=['leaper'])
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
        ctrl = ScriptedCPGControllerLeaper(env, robot=args.env, preset=args.preset)
        ep = collect_episode(env, ctrl, args.max_ep_len, seed=args.seed)
        env.close()
        print(f"[cpg] {args.env} ({args.preset}): {len(ep['actions'])} steps, "
              f"return={np.sum(ep['rewards']):.2f}")

if __name__ == '__main__':
    main()