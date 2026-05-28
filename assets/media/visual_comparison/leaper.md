# Leaper — Visual Comparison Clips

This folder contains the full evaluation clips for the **qualitative animation comparison** on the `Leaper` environment (the 12-DoF quadruped with three-link legs), shown below as animated GIFs. They accompany the visual comparison reported in the ARC-RL chapter of the dissertation, and show in motion what the static frames in the thesis figure can only hint at.

> **Note.** Each clip is a single representative evaluation rollout, recorded with deterministic actions at the end of training. As stated in the dissertation, these sequences are illustrative and should **not** be generalised to all evaluation rollouts or across all environments.

## How to read these clips

All six policies are trained against the **same** multi-objective reward, which encodes a target forward speed and a **diagonal-trot** contact pattern. The goal of the comparison is *stylistic fidelity*: not just whether the agent moves forward, but whether it moves the way the reward intends — correct heading, upright posture, and a clean diagonal trot.

A useful visual anchor: the round black **"eye"** on the front of the chassis marks the intended forward-facing direction. Watch whether each agent actually travels in the direction its eye points.

The clips are organised in two groups, matching the two rows of the thesis figure.

## Online solutions (no prior data)

### SAC
![SAC on Leaper](sac_leaper.gif)

Fails to hold the forward heading — the model rotates and moves **laterally** rather than in the eye-forward direction.

### SPEQ
![SPEQ on Leaper](speq_leaper.gif)

Better aligned with forward motion, but shows postural anomalies, most visibly an **excessive forward pitch** of the torso.

### SOPE-EO
![SOPE-EO on Leaper](sope_eo_leaper.gif)

Closest to the reference among the online methods, but still **overreaches** — the front legs extend too far forward before striking the ground.

## Online with prior data

### SACfD
![SACfD on Leaper](sacfd_leaper.gif)

Tracks the overall reference trajectory robustly, but mechanical flaws persist: **stiff rear legs** pushing the body rigidly, and front legs that **bend inward** unnaturally.

### SPEQ-O2O
![SPEQ-O2O on Leaper](speq_o2o_leaper.gif)

The prior corrects the severe forward-bending of its online counterpart; the overall stance still leaves some room for refinement.

### SOPE
![SOPE on Leaper](sope_leaper.gif)

**Highest visual fidelity** — almost perfectly replicates the reference animation, with a clean, natural diagonal trot.

## Takeaway

Comparing the two groups makes the effect of prior data immediately visible: the prior-data methods reproduce the diagonal-trot gait and the stylistic constraints far more faithfully than their online-only counterparts, with **SOPE** producing the closest match to the intended animation.

## Reference

These clips correspond to the *Visual comparison of the generated animations* subsection of the ARC-RL chapter. See the dissertation for the full experimental setup, reward definition, and quantitative learning curves.