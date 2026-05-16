# LunarLander Experiments for PA3

This folder contains the LunarLander part of Programming Assignment 3 for
Reinforcement Learning. It includes code, configs, scripts, saved logs, and
plotting utilities for the SAC experiments in Question 2.2.

The experiments covered here are:

- Q2.2.1 and Q2.2.2: continuous SAC on `LunarLanderContinuous-v3`
- Q2.2.3: hover reward variant with fixed and automatic temperature SAC
- Q2.2.4: discrete SAC on `LunarLander-v3`
- Q2.2.4 comparison baseline: DQN on `LunarLander-v3`

The SAC implementation is based on the PyTorch SAC code structure from
`https://github.com/denisyarats/pytorch_sac`, adapted for the assignment tasks.

## Folder Structure

```text
q2_lunar_lander/
├── README.md
├── generate_report_plots.py
├── q2_2_1_continuous_sac/
│   └── pytorch_sac/
│       ├── agent/
│       ├── config/
│       ├── exp_local/
│       ├── data/
│       ├── train.py
│       ├── run_baseline.sh
│       ├── requirements.txt
│       └── conda_env.yml
├── q2_2_3_hover_reward_flip/
│   └── pytorch_sac/
│       ├── agent/
│       ├── config/
│       ├── exp_local/
│       ├── hover_wrapper.py
│       ├── train.py
│       ├── run_hover_auto.sh
│       ├── run_hover_fixed.sh
│       ├── requirements.txt
│       └── conda_env.yml
└── q2_2_4_discrete_sac_vs_dqn/
    ├── discrete_sac/
    │   ├── agent/
    │   ├── config/
    │   ├── exp_local/
    │   ├── train.py
    │   ├── run_discrete_sac.sh
    │   ├── requirements.txt
    │   └── conda_env.yml
    ├── dqn/
    │   ├── dqn_agent.py
    │   ├── train.py
    │   └── run_dqn.sh
    ├── plot_eval_comparison.py
    ├── plot_training_comparison.py
    └── plot_mixed_comparison.py
```

The `exp_local/` folders contain saved training and evaluation logs from the
runs. The generated report plots are saved in the root-level `images/` folder.

## Environment Details

The main environments used are:

- `LunarLanderContinuous-v3` for continuous SAC
- `LunarLanderContinuous-v3` with a custom hover reward wrapper for Q2.2.3
- `LunarLander-v3` for discrete SAC and DQN

Important assignment settings:

- Discount factor: `0.99`
- Initial random exploration for SAC: `10000` environment steps
- Offline evaluation frequency: every `10000` environment steps
- Evaluation episodes: `20`
- Number of seeds used for final experiments: `15`

Each SAC folder includes both `requirements.txt` and `conda_env.yml`. A typical
setup is:

```bash
cd q2_lunar_lander/q2_2_1_continuous_sac/pytorch_sac
conda env create -f conda_env.yml
conda activate pytorch_sac
```

If using `pip`, install the required packages from the relevant
`requirements.txt` file inside the experiment folder.

## How to Run

Run commands from inside the corresponding experiment folder.

### Continuous SAC Baseline

```bash
cd q2_lunar_lander/q2_2_1_continuous_sac/pytorch_sac
bash run_baseline.sh
```

Single seed example:

```bash
python train.py env=LunarLanderContinuous-v3 seed=1 experiment=baseline
```

### Hover Reward Flip Experiment

Automatic temperature SAC:

```bash
cd q2_lunar_lander/q2_2_3_hover_reward_flip/pytorch_sac
bash run_hover_auto.sh
```

Fixed temperature SAC:

```bash
cd q2_lunar_lander/q2_2_3_hover_reward_flip/pytorch_sac
bash run_hover_fixed.sh
```

The hover wrapper gives a one-time hover reward when the lander enters the box
`|x| < 0.1` and `0.4 < |y| < 0.6`. During training, the reward changes from
`+200` to `-100` at the configured switch point.

### Discrete SAC

```bash
cd q2_lunar_lander/q2_2_4_discrete_sac_vs_dqn/discrete_sac
bash run_discrete_sac.sh
```

### DQN Baseline

```bash
cd q2_lunar_lander/q2_2_4_discrete_sac_vs_dqn/dqn
bash run_dqn.sh
```

## Plot Generation

The report plots can be regenerated from the saved CSV logs with:

```bash
python3 q2_lunar_lander/generate_report_plots.py
```

This script reads the saved logs under the experiment folders and writes plots
to:

```text
images/
```

The plots use mean returns with confidence intervals across seeds.

## Notes on Saved Logs

The repository includes saved logs from completed and partial runs. The discrete
SAC and DQN experiments have full evaluation logs for all 15 seeds.

For the continuous SAC baseline and some hover runs, older logs may have missing
offline evaluation rows because evaluation was originally tied to episode
boundaries. The training code has been updated so future runs trigger evaluation
by timestep using `_next_eval_step`.

When evaluation logs are incomplete, the plotting script uses the available
saved CSV data and avoids synthetic or manually created return values.

## Files Not Needed for Submission

The final zip should exclude local cache and environment files such as:

- `__pycache__/`
- `.pyc` files
- `.venv/`
- `.git/`
- local tokens or private files

These files are not required to reproduce the experiments.
