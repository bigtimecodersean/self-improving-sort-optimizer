# Teach a Simulated Arm to Reach

A hands-on exploration of reinforcement learning for simulated robotics — from a 2-joint arm touching a target to a four-legged ant learning to walk.

## What This Is

This project trains RL agents on [MuJoCo](https://mujoco.org/) physics environments using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). It starts simple (a 2-joint arm reaching a dot) and escalates to locomotion (a cheetah running, an ant walking). Along the way, it explores how algorithm choice and reward shaping fundamentally change what an agent learns.

## Setup

```bash
python -m venv reacher-env
source reacher-env/bin/activate
pip install -r requirements.txt
```

## Project Structure

Scripts are numbered in the order you should run them:

| Script | What it does |
|--------|-------------|
| `01_explore.py` | Watch random actions on Reacher — the "before" baseline |
| `02_train_ppo.py` | Train PPO on Reacher-v5 (500k steps) |
| `03a_train_sac.py` | Train SAC on Reacher-v5 for algorithm comparison |
| `03b_train_ppo_smooth.py` | Train PPO with jerkiness penalty (reward shaping) |
| `03c_train_ppo_sparse.py` | Train PPO with sparse binary reward |
| `04_enjoy_reacher.py` | Watch any trained Reacher agent (edit model path inside) |
| `05_fair_compare.py` | Evaluate all agents on the same reward for fair comparison |
| `06_plot_curves.py` | Plot learning curves for all experiments |
| `07_train_cheetah.py` | Train SAC on HalfCheetah-v5 (1M steps, ~1hr) |
| `08_train_ant.py` | Train PPO on Ant-v5 (500k steps) |
| `09_enjoy_progression.py` | Replay saved checkpoints to watch learning progression |

## Key Findings

### SAC outperformed PPO on Reacher

SAC learned faster and reached higher final reward on the same task, thanks to its replay buffer making better use of past experience. The code change was trivial — `PPO` → `SAC`.

### Reward scale ≠ reward quality

The sparse-reward agent appeared to score highest on the learning curve plot (~0 vs. -5 for SAC) but behaved the worst. It was using a different reward scale (0 or 1) than the others (negative distance + control cost). Comparing reward numbers across different reward functions is meaningless — you have to watch the behavior or evaluate on a common metric.

### Reward shaping is the real problem

The smooth (jerkiness penalty) agent solved the same task but moved differently — trading some speed for grace. The sparse agent barely learned at all because it got no signal until accidentally touching the target. The algorithm is almost secondary to what you choose to reward.

### Agents exploit, not solve

The HalfCheetah discovered that running on its back was a perfectly valid high-reward strategy because the reward function (forward velocity - control cost) says nothing about orientation. Quantitative improvement (reward 1200 → 2000) happened without qualitative behavior change — it just got better at the same exploit. This is reward hacking, and it's one of the central unsolved problems in embodied AI.

## Progression Replay

The most satisfying part of this project is watching agents improve over training. The checkpoint scripts save snapshots every 100k steps so you can replay them in order:

```bash
python 09_enjoy_progression.py ant       # watch ant go from wobbling to walking
python 09_enjoy_progression.py cheetah   # watch cheetah get faster
```

## What I'd Try Next

- **Make sparse reward work** using Hindsight Experience Replay (HER) or curiosity-driven exploration
- **Train Ant longer** (2M+ steps) to see a real gait emerge
- **Try Humanoid-v5** — full bipedal walking, the hardest standard MuJoCo task
- **Sim-to-real** — look into Isaac Gym for GPU-accelerated training and physical robot transfer

## Dependencies

- Python 3.9+
- gymnasium[mujoco]
- stable-baselines3
- matplotlib
