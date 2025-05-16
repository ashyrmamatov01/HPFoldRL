"""agents/tabular_q.py
Advanced Tabular Q(λ) agent for HP2DEnv.
Improvements over vanilla Q‑learning:
  • Linear OR visit‑based α‑decay.
  • Two exploration schemes: ε‑greedy (default) **or** UCB1.
  • Eligibility traces (replacing) with configurable λ.
  • Episode‑length guard (caller can stop at max_steps).

All features remain orthogonal – you can switch strategies via kwargs.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from math import sqrt, log
from typing import Dict, Tuple

import numpy as np

from hp_problem.utils import set_seed

logger = logging.getLogger(__name__)

StateKey = bytes  # hashable grid state


class TabularQAgent:
    """Tabular Q(λ) agent with ε‑greedy **or** UCB exploration and α‑decay.

    Parameters
    ----------
    n_actions : intuu
        Size of discrete action space.
    gamma : float, default 1.0
        Discount factor (episodic, so usually 1.0).
    alpha : float, default 0.5
        Initial learning‑rate. Can decay each visit if `alpha_decay=True`.
    alpha_decay : bool, default False
        If True, uses 1/(1+N(s,a)) for step‑size (Robbins‑Monro).
    lam : float, default 0.0
        Eligibility trace parameter. 0 ⇒ plain Q‑learning. 1 ⇒ Monte‑Carlo.
    exploration : {"eps", "ucb"}, default "eps"
        Exploration strategy.
    eps_start / eps_end / eps_decay_steps : float / float / int
        ε‑greedy schedule.
    ucb_c : float, default 1.0
        Exploration constant for UCB.
    optimistic_init : float, default 0.0
        Initial Q for unseen (s,a).
    seed : int | None
    """

    def __init__(
        self,
        n_actions: int,
        *,
        gamma: float = 1.0,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        lam: float = 0.0,
        exploration: str = "eps",
        eps_start: float = 0.99,
        eps_end: float = 0.05,
        # eps_decay_steps: int = 10_000,
        ucb_c: float = 1.0,
        optimistic_init: float = 0.0,
        seed: int | None = None,
        eps_decay_frac_warmup: float = 0.05, # e.g., 5% of total steps for warmup
        eps_decay_frac_reach_min: float = 0.95, # e.g., reach eps_end at 95% of total steps
        total_episodes_for_decay: int = 10000, # Pass total episodes from training script
        avg_steps_per_episode: int = 20,    # Pass avg episode length (n_res - 1)
    ) -> None:
        assert exploration in {"eps", "ucb"}
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha0 = alpha
        self.alpha_decay = alpha_decay
        self.lam = lam
        self.exploration = exploration
        # self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_agent_steps_for_decay_calc = total_episodes_for_decay * avg_steps_per_episode
        self.eps_warmup_steps = int(eps_decay_frac_warmup * self.total_agent_steps_for_decay_calc)
        # Steps over which decay actually happens
        decay_phase_duration = int((eps_decay_frac_reach_min - eps_decay_frac_warmup) * self.total_agent_steps_for_decay_calc)
        # Ensure decay_phase_duration is positive
        self.eps_decay_steps_for_slope = max(1, decay_phase_duration) 

        self.eps_reach_min_step = self.eps_warmup_steps + self.eps_decay_steps_for_slope 
        
        
        self.ucb_c = ucb_c
        self.optimistic_init = optimistic_init

        self.rng = np.random.default_rng(set_seed(seed))

        # Main tables
        self.Q: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.full(n_actions, optimistic_init, dtype=np.float32)
        )
        self.E: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )  # eligibility traces
        self.N: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.int32)
        )  # visit counts for α‑decay / UCB

        self.global_step = 0  # for eps schedule & UCB

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _state_key(self, obs: np.ndarray) -> StateKey:
        return obs.tobytes()

    # def _epsilon(self) -> float:
    #     frac = min(1.0, self.global_step / self.eps_decay_steps)
    #     return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _epsilon(self, step_idx: int | None = None) -> float:
        """Compute the current epsilon value based on the global step."""
        idx = self.global_step if step_idx is None else step_idx
        # Calculate the total number of steps for decay
        if idx < self.eps_warmup_steps:
            return self.eps_start
        elif idx >= self.eps_reach_min_step:
            return self.eps_end
        else:
            # Linear decay during the decay phase
            # Number of steps into the decay phase
            current_decay_step = idx - self.eps_warmup_steps
            frac = min(1.0, current_decay_step / self.eps_decay_steps_for_slope)
            return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _alpha(self, s: StateKey, a: int) -> float:
        if not self.alpha_decay:
            return self.alpha0
        # 1 / (1 + visits)
        visits = self.N[s][a]
        return 1.0 / (1.0 + visits)

    # ------------------------------------------------------------------ #
    # action selection
    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray, episode: int = None) -> int:
        key = self._state_key(obs)
        q = self.Q[key]
        self.global_step += 1

        if self.exploration == "eps":
            eps = self._epsilon(episode)
            # if no valid actions, pick any random action (→ env will terminate)
            if not valid_mask.any():
                return int(self.rng.integers(self.n_actions))

            # ε-explore
            if self.rng.random() < eps:
                return int(self.rng.choice(np.flatnonzero(valid_mask)))

            # greedy among the valid
            masked_q = np.where(valid_mask, q, -np.inf)
            best = np.flatnonzero(masked_q == masked_q.max())
            if best.size == 0:
                # fallback to any valid
                return int(self.rng.choice(np.flatnonzero(valid_mask)))
            return int(self.rng.choice(best))

        # ----------------------------------------------------------------
        # UCB‐1 fallback (same guards as ε-greedy)
        assert self.exploration == "ucb"
        if not valid_mask.any():
            return int(self.rng.integers(self.n_actions))
        total = max(1, self.global_step)
        counts = self.N[key]
        ucb_bonus = self.ucb_c * np.sqrt(np.log(total) / (1e-8 + counts))
        masked_val = np.where(valid_mask, q + ucb_bonus, -np.inf)
        best = np.flatnonzero(masked_val == masked_val.max())
        if best.size == 0:
            return int(self.rng.choice(np.flatnonzero(valid_mask)))
        return int(self.rng.choice(best))

    # ------------------------------------------------------------------ #
    # Q(λ) TD update
    # ------------------------------------------------------------------ #
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_valid_mask: np.ndarray,
    ) -> None:
        s, ns = self._state_key(obs), self._state_key(next_obs)

        # delta: if there are truly no next‐moves, treat as terminal
        target = reward
        if not done and next_valid_mask.any():
            # Q[ns] stores expected future *rewards* (which are positive for good outcomes)
            # target += self.gamma * np.max(
            #     np.where(next_valid_mask, self.Q[ns], -np.inf)
            # )
            max_next_q = np.max(np.where(next_valid_mask, self.Q[ns], -np.inf))
            if max_next_q > -np.inf: # Avoid adding -inf if all next Qs are -inf
                target += self.gamma * max_next_q
        delta = target - self.Q[s][action]

        # update eligibilities (replacing trace)
        # decay all traces
        for key in list(self.E.keys()):
            self.E[key] *= self.gamma * self.lam
        # set trace for current (s,a)
        self.E[s][action] = 1.0

        # Q(λ) updates only where the trace is nonzero
        for key, e_vec in self.E.items():
            nz = e_vec != 0.0
            if not nz.any():
                continue
            _alpha = self._alpha(key, action)
            # only multiply the nonzero entries to avoid 0 * ±inf → NaN
            self.Q[key][nz] += _alpha * delta * e_vec[nz] # Q-values will learn to predict these transformed rewards

        # visit counts (for α‑decay / UCB)
        self.N[s][action] += 1

    # ------------------------------------------------------------------ #
    # greedy helper for evaluation
    # ------------------------------------------------------------------ #
    def greedy(self, obs: np.ndarray, valid_mask: np.ndarray) -> int:
        key = self._state_key(obs)
        masked = np.where(valid_mask, self.Q[key], -np.inf)
        best = np.flatnonzero(masked == masked.max())
        return int(self.rng.choice(best))

