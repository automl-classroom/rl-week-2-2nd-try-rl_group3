from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # type: ignore[import]
from rich import print as printr


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        n_states = 2
        n_actions = 2
        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.horizon = 13

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # ChatGPT hat geholfen
        self.state = 0
        return self.state, {}

    def step(self, action):
        self.state = action

        reward = action

        terminated = True
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def get_reward_per_action(self):
        reward_matrix = np.array([[0, 1], [0, 1]])

        return reward_matrix

    def get_transition_matrix(self):
        transition_matrix = np.array(
            [
                [[1, 0], [0, 1]],  # state 0
                [[1, 0], [0, 1]],
            ]
        )  # state 1

        return transition_matrix


# mars rover
env = MyEnv()
actions = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]

states = []
s, info = env.reset()
states.append(s)
for i in range(env.horizon):
    action = actions[i]
    s_next, r, terminated, truncated, info = env.step(action)
    printr(
        f"Step: {i}, state: {s}, action: {action}, next state: {s_next}, "
        f"reward: {r}, terminated: {terminated}, truncated: {truncated}"
    )
    s = s_next
    states.append(s)

# Plot
fig, ax = plt.subplots()
image = plt.imread("figures/alien_1f47d.png")
image_box = OffsetImage(image, zoom=0.1)
x = np.arange(0, len(states))
y = states
for x0, y0 in zip(x, y):
    ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
    ax.add_artist(ab)
ax.plot(x, y, c="green")
ax.set_xlabel("Step")
ax.set_ylabel("State")
plt.show()


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
