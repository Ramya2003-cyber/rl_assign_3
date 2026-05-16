

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite



def _get_physics_quantities(physics) -> dict:
    """Extract raw physics quantities needed for all reward computations."""
    finger_pos = physics.named.data.geom_xpos['finger', :2].copy()
    target_pos = physics.named.data.geom_xpos['target', :2].copy()
    finger_vel = physics.named.data.geom_xpos['finger', :2].copy()
    joint_vel = physics.data.qvel[:].copy()
    return {
        'finger_pos': finger_pos,
        'target_pos': target_pos,
        'joint_vel': joint_vel,
    }


def _compute_distance(physics) -> float:
    """Euclidean distance between fingertip and target in the 2-D task plane."""
    q = _get_physics_quantities(physics)
    return float(np.linalg.norm(q['finger_pos'] - q['target_pos']))


def _in_target(dm_env, physics) -> bool:
   
    try:
        return bool(dm_env.task.within_target(physics))
    except AttributeError:
        return _compute_distance(physics) < 0.05




class ReacherWrapper(gym.Env):
   

    _VEL_THRESHOLD: float = 0.05       # near-zero velocity threshold for Rc termination
    _DIST_THRESHOLD: float = 0.05
    _MAX_EPISODE_STEPS: int = 1000
    _TIMEOUT_PENALTY: float = -20.0

    def __init__(self, reward_type: str = 'a', seed: int = 0):
        super().__init__()
        assert reward_type in ('a', 'b', 'c'), \
            f"reward_type must be 'a', 'b', or 'c'; got '{reward_type}'"

        self.reward_type = reward_type
        self._seed = seed

        self._dm_env = suite.load(
            domain_name='reacher',
            task_name='easy',
            task_kwargs={'random': seed,
            'time_limit': float('inf')
            },
        )
        self._physics = self._dm_env.physics

        obs_spec = self._dm_env.observation_spec()
        obs_dim = sum(int(np.prod(v.shape)) for v in obs_spec.values())

        action_spec = self._dm_env.action_spec()
        action_dim = int(np.prod(action_spec.shape))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        self._step_count: int = 0
        self._timeout_count: int = 0
        self._last_obs: np.ndarray | None = None
        self._last_action: np.ndarray | None = None



    def _flatten_obs(self, time_step) -> np.ndarray:
        """Flatten all dm_control observation arrays into a single vector."""
        parts = [np.atleast_1d(v).ravel() for v in time_step.observation.values()]
        return np.concatenate(parts).astype(np.float32)

    def _arm_in_target(self) -> bool:
        return _in_target(self._dm_env, self._physics)

    def _arm_velocity(self) -> float:
        """Scalar norm of all joint velocities (proxy for arm speed)."""
        return float(np.linalg.norm(self._physics.data.qvel[:]))

    def _compute_all_rewards(self, action: np.ndarray) -> dict[str, float]:
        """
        Compute all three reward values for a *given action* and the *current*
        physics state.  Used by the cross-evaluation protocol.

        Returns a dict with keys ``'a'``, ``'b'``, ``'c'``.
        """
        in_target = self._arm_in_target()
        dist = _compute_distance(self._physics)
        action_penalty = float(np.dot(action, action))  # ||a||^2

        ra = 1.0 if in_target else -dist - action_penalty
        rb = 1.0 if in_target else 0.0
        rc = -1.0  # always -1 per step for Rc

        return {'a': ra, 'b': rb, 'c': rc}

    def _partial_reset_arm(self):

        with self._physics.reset_context():
            target_qpos = self._physics.data.qpos[2:].copy()
            self._physics.data.qpos[:2] = self._dm_env.task.random.uniform(
                low=-np.pi, high=np.pi, size=2
            )
            self._physics.data.qvel[:2] = 0.0
            self._physics.data.qpos[2:] = target_qpos

    def _obs_from_physics(self) -> np.ndarray:

        obs_dict = self._dm_env.task.get_observation(self._physics)
        parts = [np.atleast_1d(v).ravel() for v in obs_dict.values()]
        return np.concatenate(parts).astype(np.float32)


    def reset(self, *, seed=None, options=None):
        """Full episode reset."""
        if seed is not None:
            self._seed = seed

        self._step_count = 0
        self._timeout_count = 0

        time_step = self._dm_env.reset()
        obs = self._flatten_obs(time_step)
        self._last_obs = obs
        return obs, {}

    def step(self, action: np.ndarray):
        """Step the environment; returns (obs, reward, terminated, truncated, info)."""
        self._last_action = action
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        time_step = self._dm_env.step(action_clipped)
        obs = self._flatten_obs(time_step)
        self._last_obs = obs
        self._step_count += 1

        all_rewards = self._compute_all_rewards(action_clipped)
        in_target = self._arm_in_target()

        if self.reward_type == 'c':
            reward = all_rewards['c']
            arm_vel = self._arm_velocity()
            terminal = in_target and (arm_vel < self._VEL_THRESHOLD)

            if terminal:
                terminated = True
                truncated = False
                info = {'in_target': True, 'timeout': False, 'all_rewards': all_rewards}
            elif self._step_count % self._MAX_EPISODE_STEPS == 0:
                # Timeout: apply penalty + partial reset, episode continues
                reward += self._TIMEOUT_PENALTY
                self._timeout_count += 1
                self._partial_reset_arm()
                obs = self._obs_from_physics()
                self._last_obs = obs
                terminated = False
                truncated = False
                info = {
                    'in_target': False,
                    'timeout': True,
                    'timeout_penalty': self._TIMEOUT_PENALTY,
                    'timeout_count': self._timeout_count,
                    'all_rewards': all_rewards,
                }
            else:
                terminated = False
                truncated = False
                info = {'in_target': in_target, 'timeout': False, 'all_rewards': all_rewards}

        else:
            reward = all_rewards[self.reward_type]
            terminated = False
            truncated = self._step_count >= self._MAX_EPISODE_STEPS
            info = {'in_target': in_target, 'timeout': False, 'all_rewards': all_rewards}

        return obs, reward, terminated, truncated, info

    def compute_all_rewards(self, action: np.ndarray) -> dict[str, float]:
        """
        Public method for the cross-evaluation protocol.
        Call *after* a step to get all three reward values for that transition.
        """
        return self._compute_all_rewards(action)

    def get_target_pos(self) -> np.ndarray:
        """Return the 2-D position of the target (used in Rc partial reset)."""
        return self._physics.named.data.geom_xpos['target', :2].copy()

    def render(self, mode='rgb_array'):
        return self._physics.render(camera_id=0)

    def close(self):
        pass

    @property
    def max_episode_steps(self) -> int:
        return self._MAX_EPISODE_STEPS
