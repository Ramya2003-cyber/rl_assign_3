"""
envs/reacher_custom.py
======================
Gym-like wrapper around dm_control's reacher (easy) environment.

Supports three reward formulations:
  Ra  – distance penalty      (truncated at 1000 steps, no early termination)
  Rb  – sparse binary reward  (truncated at 1000 steps, no early termination)
  Rc  – step penalty          (episode terminates only on reaching target with
                               near-zero velocity; timeout at 1000 steps applies
                               a -20 penalty and partially resets the arm while
                               keeping the target fixed — the episode continues)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite


# ---------------------------------------------------------------------------
# Helpers for extracting structured physics quantities from dm_control
# ---------------------------------------------------------------------------

def _get_physics_quantities(physics) -> dict:
    """Extract raw physics quantities needed for all reward computations."""
    # Fingertip (end-effector) position in 2-D task plane (x, y)
    finger_pos = physics.named.data.geom_xpos['finger', :2].copy()
    # Target position in 2-D task plane
    target_pos = physics.named.data.geom_xpos['target', :2].copy()
    # Velocity of the fingertip (for Rc terminal condition)
    finger_vel = physics.named.data.geom_xpos['finger', :2].copy()  # placeholder
    # Use joint velocities as a proxy for arm velocity
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
    """
    Use the dm_control task's own `within_target` method when available,
    otherwise fall back to a distance threshold.
    """
    try:
        return bool(dm_env.task.within_target(physics))
    except AttributeError:
        return _compute_distance(physics) < 0.05


# ---------------------------------------------------------------------------
# ReacherWrapper
# ---------------------------------------------------------------------------

class ReacherWrapper(gym.Env):
    """
    Gym-compatible wrapper for dm_control ``reacher`` (easy).

    Parameters
    ----------
    reward_type : str
        One of ``'a'``, ``'b'``, or ``'c'``.
    seed : int, optional
        Random seed for the underlying dm_control environment.
    """

    # Threshold used by Rc to decide "near-zero velocity" for termination.
    _VEL_THRESHOLD: float = 0.05
    # Radius threshold for "in-target" when the task does not expose its own.
    _DIST_THRESHOLD: float = 0.05
    # Maximum steps per timeout window for Ra / Rb (and Rc's partial-reset trigger).
    _MAX_EPISODE_STEPS: int = 1000
    # Timeout penalty applied by Rc.
    _TIMEOUT_PENALTY: float = -20.0

    def __init__(self, reward_type: str = 'a', seed: int = 0):
        super().__init__()
        assert reward_type in ('a', 'b', 'c'), \
            f"reward_type must be 'a', 'b', or 'c'; got '{reward_type}'"

        self.reward_type = reward_type
        self._seed = seed

        # ── Build dm_control environment ────────────────────────────────────
        self._dm_env = suite.load(
            domain_name='reacher',
            task_name='easy',
            task_kwargs={'random': seed},
        )
        self._physics = self._dm_env.physics

        # ── Derive Gym spaces from dm_control specs ──────────────────────────
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

        # Episode counters (public so the training loop can read them)
        self._step_count: int = 0          # steps since last *full* reset
        self._timeout_count: int = 0       # number of Rc timeouts in this episode
        # Store current obs so we can compute the cross-eval reward outside step()
        self._last_obs: np.ndarray | None = None
        self._last_action: np.ndarray | None = None

    # ── Internal helpers ────────────────────────────────────────────────────

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
        """
        Rc partial-reset: randomise the arm joint angles / velocities while
        keeping the target in exactly the same position.

        dm_control stores the target position in `self._physics.named.data.qpos`.
        The reacher model uses:
          qpos[0:2]  – arm joints (angle1, angle2)
          qpos[2:4]  – target position (x, y) expressed as joint-space offsets
        We resample only qpos[0:2] and qvel[0:2].
        """
        with self._physics.reset_context():
            # Preserve target qpos (indices 2 and 3 in the reacher model)
            target_qpos = self._physics.data.qpos[2:].copy()
            # Randomise arm joints uniformly
            self._physics.data.qpos[:2] = self._dm_env.task.random.uniform(
                low=-np.pi, high=np.pi, size=2
            )
            # Zero arm velocities (clean start for the new attempt)
            self._physics.data.qvel[:2] = 0.0
            # Restore target
            self._physics.data.qpos[2:] = target_qpos

    def _obs_from_physics(self) -> np.ndarray:
        """
        Build an observation vector from the *current* physics state without
        calling ``dm_env.step`` by dynamically asking the dm_control task.
        This guarantees the shape exactly matches the standard observation.
        """
        obs_dict = self._dm_env.task.get_observation(self._physics)
        parts = [np.atleast_1d(v).ravel() for v in obs_dict.values()]
        return np.concatenate(parts).astype(np.float32)
    # ── Gym interface ───────────────────────────────────────────────────────

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
        """
        Step the environment.

        Returns the standard 5-tuple:
            obs, reward, terminated, truncated, info

        The ``done`` flag passed to the replay buffer should be
        ``terminated or truncated``.

        For Rc, ``truncated`` is *never* True from here (the episode continues
        after a timeout).  The training loop must handle the case that it has
        run for 1000 steps on the current episode without success, but from the
        perspective of the replay buffer each step is a normal transition.
        """
        self._last_action = action
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        time_step = self._dm_env.step(action_clipped)
        obs = self._flatten_obs(time_step)
        self._last_obs = obs
        self._step_count += 1

        all_rewards = self._compute_all_rewards(action_clipped)
        in_target = self._arm_in_target()

        # ── Rc logic ──────────────────────────────────────────────────────
        if self.reward_type == 'c':
            reward = all_rewards['c']  # -1 per step
            arm_vel = self._arm_velocity()
            terminal = in_target and (arm_vel < self._VEL_THRESHOLD)

            if terminal:
                # True episode termination — agent reached goal with near-zero vel
                terminated = True
                truncated = False
                info = {'in_target': True, 'timeout': False, 'all_rewards': all_rewards}
            elif self._step_count % self._MAX_EPISODE_STEPS == 0:
                # ── Timeout: apply penalty + partial reset, episode continues ──
                reward += self._TIMEOUT_PENALTY
                self._timeout_count += 1
                self._partial_reset_arm()
                # Re-derive next-obs from current (post-reset) physics state.
                # We do NOT issue another dm_env.step; instead we build the
                # observation by reading the physics quantities directly,
                # which is equivalent to the observation the agent would see
                # at the start of the next sub-episode.
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

        # ── Ra / Rb logic ──────────────────────────────────────────────────
        else:
            reward = all_rewards[self.reward_type]
            terminated = False  # no early termination for Ra / Rb
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

    # Expose max episode steps so the training loop can access it cleanly
    @property
    def max_episode_steps(self) -> int:
        return self._MAX_EPISODE_STEPS
