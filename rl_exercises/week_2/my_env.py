from __future__ import annotations
import numpy as np
import gymnasium as gym


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

    """Initializes the observation and action space for the environment."""
    def __init__(self, 
             rewards: np.ndarray = np.array([0, 1]),
             horizon: int = 10,
             seed: int | None = None):
             
             self.rewards = list(rewards)
             self.horizon = int(horizon)
             self.current_steps = 0
             self.position = 0
             # spaces
             self.observation_space = gym.spaces.Discrete(2) # single number between 0 and n - 1
             self.action_space = gym.spaces.Discrete(2) #agent can take actions 0 or 1

             # helpers
             self.states = np.arange(2) 
             self.actions = np.arange(2)

             # transition matrix
             self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
              self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ) -> tuple[int, dict[str, Any]]:
          
          self.current_steps = 0
          self.position = 0
          return self.position, {}

    def step(self, 
             action: int
             ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        action : int
            Action to take (0: move to state 0, 1: move to state 1).
        """
        action = int(action)
        if not self.action_space.contains(action): #self.action_space = gym.spaces.Discrete(2)
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        self.position = action # go to the state indicated by the action
        reward = int(self.rewards[action])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the reward function R[s, a] for each (state (s), action(a)) pair.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float) #R=Reward
        for s in range(nS):
            for a in range(nA):
                R[s, a] = float(self.rewards[a])
        return R

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a deterministic transition matrix T[s, a, s'].
        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor of transition probabilities
        """
        if S is None or A is None:
            S, A, P = self.states, self.actions

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in S:
            for a in A:
                s_next = a
                T[s, a, s_next] = 1.0
        return T

    def render(self, mode: str = "human"):
        print(f"[MyEnv] pos={self.position}, steps={self.current_steps}")     


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
