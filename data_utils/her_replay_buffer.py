"""
Hindsight Experience Replay (HER) buffer for goal-conditioned SAC.

Extends the FIFO ReplayBuffer with episode-level staging and hindsight
relabeling. On episode end, original transitions plus k relabeled copies
(using the "future" strategy of Andrychowicz et al., 2017) are flushed
to the main storage and become available for uniform sampling.

The relabel step replaces the desired_goal in each transition with a
goal that was actually achieved later in the same episode, then recomputes
the reward via the env's compute_reward function. This turns failed
episodes into successful ones for synthetic goals, providing dense
learning signal in sparse-reward goal-conditioned tasks.

Sample composition after flushing an episode of length T (with k relabels
per transition, last transition has no future and is not relabeled):
    - T original transitions
    - (T - 1) * k relabeled transitions
Uniform sampling from the main storage therefore yields roughly
1 / (1 + k) originals and k / (1 + k) relabels, after the buffer has
warmed up to steady state.

Reference:
    Andrychowicz et al. (2017), "Hindsight Experience Replay"
"""

import numpy as np
import torch

from data_utils.replay_buffer import ReplayBuffer


class HERReplayBuffer(ReplayBuffer):
    """
    Episode-aware FIFO replay buffer with hindsight relabeling.

    Storage layout (parent ReplayBuffer arrays) extended with:
        next_achieved_goal: (capacity, goal_dim) float32
            The achieved_goal corresponding to next_obs of each transition.
            Used during relabel as both (a) the input to compute_reward
            against the synthetic goal and (b) the source pool of
            synthetic goals for earlier timesteps in the same episode.

    Workflow:
        1. add(...) appends a transition to an in-memory episode buffer.
           Nothing is written to main storage yet.
        2. flush_episode() is called when the episode ends. It writes
           the originals plus k relabeled copies of each transition
           (except the last) to the main FIFO storage.
        3. sample() is inherited from ReplayBuffer unchanged.

    Args:
        capacity: max transitions in main storage.
        obs_dim, action_dim: dimensions (same as parent).
        goal_dim: dimension of achieved_goal / desired_goal (3 for Fetch).
        goal_slice: slice describing where desired_goal sits in the flat
            observation. For the Fetch wrapper this is slice(25, 28).
        compute_reward_fn: callable(achieved_goal, desired_goal, info) -> float.
            Should match env.compute_reward semantics. Supports scalar
            or batched inputs; this buffer calls it per-transition for
            clarity (50 transitions * 4 relabels = 200 calls per episode,
            not a bottleneck on this scale).
        k: number of relabeled copies per original transition.
        strategy: "future" only. Other strategies ("final", "episode",
            "random") are easy to add later if needed.
        device: torch device for sampled batches.
    """

    def __init__(self, capacity, obs_dim, action_dim, goal_dim,
                 goal_slice, compute_reward_fn,
                 k=4, strategy="future", device="cpu"):
        super().__init__(capacity, obs_dim, action_dim, device=device)

        if strategy != "future":
            raise NotImplementedError(
                f"HER strategy '{strategy}' not implemented; only 'future' is "
                "available. Add new strategies in _sample_relabel_indices()."
            )

        self.goal_dim = goal_dim
        self.goal_slice = goal_slice
        self.compute_reward = compute_reward_fn
        self.k = k
        self.strategy = strategy

        # Additional per-transition storage: the achieved_goal AT next_obs.
        # Used during relabel to (a) compute reward against synthetic goals
        # and (b) serve as the pool of synthetic goals for earlier timesteps.
        self.next_achieved_goal = np.zeros((capacity, goal_dim), dtype=np.float32)

        # Episode-in-progress staging area. Each element is a dict with the
        # same fields as main storage. Cleared on flush_episode().
        self._episode = []

    def add(self, obs, action, reward, next_obs, done, next_achieved_goal):
        """
        Append a transition to the *episode* buffer. Does NOT touch the
        main FIFO storage; call flush_episode() at episode end.

        Args:
            obs, action, reward, next_obs, done: standard SAC transition.
            next_achieved_goal: 3-D array, the achieved_goal AFTER taking
                action (i.e., the achieved_goal corresponding to next_obs).
                Comes from env's info["achieved_goal"] post-step.
        """
        self._episode.append({
            "obs": np.asarray(obs, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "done": float(done),
            "next_achieved_goal": np.asarray(next_achieved_goal, dtype=np.float32),
        })

    def flush_episode(self):
        """
        Flush the staged episode to main FIFO storage.

        Two passes:
            1. Write every original transition as-is.
            2. For each timestep t in [0, T-1), sample k future indices
               j in [t+1, T) and write k relabeled copies of transition t,
               each using next_achieved_goal[j] as the synthetic goal.

        The last transition (t = T-1) has no future timesteps and is not
        relabeled.
        """
        T = len(self._episode)
        if T == 0:
            return

        # Pass 1: originals.
        for tr in self._episode:
            self._write(tr)

        # Pass 2: future-strategy relabels.
        for t in range(T - 1):  # last timestep has no future, skip
            future_indices = np.random.randint(t + 1, T, size=self.k)
            for j in future_indices:
                self._write_relabel(t, j)

        self._episode.clear()

    def _write(self, transition):
        """Write a single transition dict into the main FIFO storage."""
        i = self.pos
        self.obs[i] = transition["obs"]
        self.action[i] = transition["action"]
        self.reward[i] = transition["reward"]
        self.next_obs[i] = transition["next_obs"]
        self.done[i] = transition["done"]
        self.next_achieved_goal[i] = transition["next_achieved_goal"]
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _write_relabel(self, t, j):
        """
        Build and write one relabeled transition: original transition at t,
        with desired_goal replaced by next_achieved_goal at step j (future).

        Done flag logic:
            On the relabeled goal, the transition is "successful" when its
            recomputed reward >= 0 (sparse Fetch: reward = 0 on success,
            -1 otherwise; the relabeled (t, j=t+1, ...) transitions where
            the action lands the cube on the synthetic goal will hit this).
            Successful transitions are marked done=True to terminate Q
            bootstrap at the synthetic success state. This is consistent
            with how truncated episode ends are treated in the trainer.
        """
        orig = self._episode[t]
        new_goal = self._episode[j]["next_achieved_goal"]

        # Copy obs and next_obs, overwrite the goal slice with synthetic goal
        new_obs = orig["obs"].copy()
        new_obs[self.goal_slice] = new_goal
        new_next_obs = orig["next_obs"].copy()
        new_next_obs[self.goal_slice] = new_goal

        # Recompute reward under the new goal. Fetch's compute_reward takes
        # the achieved_goal at next_obs and the (now synthetic) desired_goal.
        new_reward = float(
            self.compute_reward(orig["next_achieved_goal"], new_goal, {})
        )

        # Sparse success heuristic: reward >= 0 means goal reached.
        # For dense reward this branch should be revisited; HER is
        # primarily a sparse-reward technique.
        success = new_reward >= 0.0
        new_done = 1.0 if success else orig["done"]

        self._write({
            "obs": new_obs,
            "action": orig["action"],
            "reward": new_reward,
            "next_obs": new_next_obs,
            "done": new_done,
            "next_achieved_goal": new_goal.copy(),
        })

    # sample() and __len__() are inherited from ReplayBuffer unchanged.
    # The relabeled transitions live in the same arrays as originals, so
    # uniform sampling naturally produces a mixed batch.