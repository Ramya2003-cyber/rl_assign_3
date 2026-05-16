# DA6400 Reinforcement Learning - Programming Assignment 3

This repository contains the code for Programming Assignment 3. The work is organized by assignment part: modified Pendulum with SAC, LunarLander SAC/DQN experiments, Reacher reward-formulation experiments, and the PEBBLE bonus experiments.

## Team Members and Contributions

- Sinigi Ramya Sri (`CS25M049`) - Part 1 Pendulum, Part 3 Reacher, and Bonus PEBBLE experiments
- Kaki Hephzi Sunanda (`DA25M015`) - Part 2 LunarLander experiments

## Repository Structure

```text
rl_assign_3/
├── README.md
├── requirements.txt
├── agent/
│   ├── actor.py
│   ├── critic.py
│   ├── sac.py
│   ├── reward_net.py
│   └── preference_buffer.py
├── envs/
│   ├── pendulum_custom.py
│   ├── reacher_custom.py
│   └── simulated_teacher.py
├── config/
│   ├── train.yaml
│   ├── reacher.yaml
│   ├── pebble_reacher.yaml
│   └── agent/sac.yaml
├── scripts/
│   ├── run_pendulum.py
│   └── run_human_render.py
├── part2_lunarlander/
│   ├── README.md
│   ├── generate_report_plots.py
│   ├── images/
│   ├── q2_2_1_continuous_sac/
│   ├── q2_2_3_hover_reward_flip/
│   └── q2_2_4_discrete_sac_vs_dqn/
├── run_2.1.py
├── run_reacher.py
├── run_pebble.py
├── run_pebble_reacher.py
├── plot_2.1.py
├── plot_q5_manual.py
├── plot_2.3.py
├── plot_q3c_cross_eval.py
└── plot_pebble_reacher.py
```

The `part2_lunarlander/` folder contains the LunarLander files that were prepared as the separate LunarLander submission package. Its own README gives the detailed LunarLander-only structure and commands.

## Environment Details

The main Python dependencies are listed in `requirements.txt`. The experiments use:

- PyTorch for SAC, reward models, actors, and critics
- Gymnasium for Pendulum and LunarLander environments
- DeepMind Control Suite for Reacher
- Hydra/OmegaConf for configuration
- NumPy, Pandas, and Matplotlib for saved results and plots

Typical setup:

```bash
cd rl_assign_3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LunarLander, use the environment files inside the specific `part2_lunarlander/` experiment folders if needed.

## Part 1: Pendulum

This part modifies `Pendulum-v1` so that the target angle can be changed. The custom environment wrapper is:

```text
envs/pendulum_custom.py
```

It uses the shaped reward

```text
-(angle_error^2 + 0.1 * angular_velocity^2 + 0.001 * action^2)
```

where the angular error is wrapped to the shortest direction around the circle.

### SAC With Automated Temperature Tuning

Run one Pendulum experiment:

```bash
python scripts/run_pendulum.py custom.target_angle=90 custom.auto_tune=true
```

Run the full Part 1 sweep script:

```bash
python run_2.1.py
```

Useful parameters:

```bash
custom.target_angle=90
custom.auto_tune=true
custom.alpha=0.1
custom.reward_scale=1.0
```

### Manual Temperature Tuning and Reward Scaling

Manual temperature tuning uses fixed values of alpha for target angles such as `-150`, `-60`, `90`, and `120`.

Example fixed-alpha run:

```bash
python scripts/run_pendulum.py custom.target_angle=90 custom.auto_tune=false custom.alpha=0.05
```

Reward scaling runs can be launched by changing `custom.reward_scale`:

```bash
python scripts/run_pendulum.py custom.target_angle=90 custom.auto_tune=true custom.reward_scale=0.1
python scripts/run_pendulum.py custom.target_angle=90 custom.auto_tune=true custom.reward_scale=10.0
```

Plot helpers:

```bash
python plot_2.1.py
python plot_q5_manual.py
```

## Part 2: Soft Actor-Critic on LunarLander

The LunarLander code is kept under:

```text
part2_lunarlander/
```

This section contains the assignment experiments:

- `2.2.1`: Continuous SAC on `LunarLanderContinuous-v3`
- `2.2.2`: Stages of SAC behavior during learning
- `2.2.3`: Hover reward modification and reward flip
- `2.2.4`: Discrete SAC on `LunarLander-v3`
- `2.2.4`: DQN baseline comparison

### 2.2.1 Continuous SAC Baseline

```bash
cd part2_lunarlander/q2_2_1_continuous_sac/pytorch_sac
bash run_baseline.sh
```

Single-seed example:

```bash
python train.py env=LunarLanderContinuous-v3 seed=1 experiment=baseline
```

### 2.2.3 Hover Reward Flip

Automatic temperature SAC:

```bash
cd part2_lunarlander/q2_2_3_hover_reward_flip/pytorch_sac
bash run_hover_auto.sh
```

Fixed temperature SAC:

```bash
cd part2_lunarlander/q2_2_3_hover_reward_flip/pytorch_sac
bash run_hover_fixed.sh
```

### 2.2.4 Discrete SAC and DQN

Discrete SAC:

```bash
cd part2_lunarlander/q2_2_4_discrete_sac_vs_dqn/discrete_sac
bash run_discrete_sac.sh
```

DQN baseline:

```bash
cd part2_lunarlander/q2_2_4_discrete_sac_vs_dqn/dqn
bash run_dqn.sh
```

Regenerate LunarLander report plots:

```bash
python part2_lunarlander/generate_report_plots.py
```

## Part 3: Reacher

This part uses the easy Reacher environment and compares three reward formulations:

- `Ra`: dense distance-based reward
- `Rb`: sparse reward for being inside the target
- `Rc`: time/velocity-based reward that requires reaching and stabilizing

The custom wrapper is:

```text
envs/reacher_custom.py
```

Run SAC for each reward formulation:

```bash
python run_reacher.py reward_type=a seed=1
python run_reacher.py reward_type=b seed=1
python run_reacher.py reward_type=c seed=1
```

The script saves files of the form:

```text
reacher_results_<reward_type>_seed_<seed>.npy
```

Plot and evaluation helpers:

```bash
python eval_q3_reacher.py
python plot_2.3.py
python plot_q3c_cross_eval.py
```

## Bonus: PEBBLE

The bonus experiments implement preference-based reward learning with a simulated teacher.

Main files:

```text
agent/reward_net.py
agent/preference_buffer.py
envs/simulated_teacher.py
run_pebble.py
run_pebble_reacher.py
reward_model.py
```

### PEBBLE on Pendulum

Example run:

```bash
python run_pebble.py custom.target_angle=90 custom.feedback_budget=1000 seed=1
```

Feedback budget experiments use values such as `100`, `500`, and `1000`.

### PEBBLE on Reacher

```bash
python run_pebble_reacher.py reward_type=a seed=1
python run_pebble_reacher.py reward_type=b seed=1
python run_pebble_reacher.py reward_type=c seed=1
```

Plot helper:

```bash
python plot_pebble_reacher.py
```
