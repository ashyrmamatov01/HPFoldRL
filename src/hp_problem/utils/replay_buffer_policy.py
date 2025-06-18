from __future__ import annotations
import random, collections, numpy as np, torch

Transition = collections.namedtuple(
    "Transition", ("obs", "prev_obs", "target_v", "target_p")
)

class ReplayBuffer:
    def __init__(
        self, 
        capacity: int, 
        obs_shape: tuple[int, ...], 
        n_actions: int,
        device: str
    ):
        self.capacity, self.device = capacity, device
        self.ptr, self.size = 0, 0
        self.obs_buf  = np.zeros((capacity, *obs_shape), dtype=np.int8)
        self.prev_obs_buf = np.zeros_like(self.obs_buf)
        self.target_v_buf = np.zeros((capacity,), dtype=np.float32)             # target value to predict from the previous state
        self.target_p_buf = np.zeros((capacity, n_actions), dtype=np.float32)  # target policy to predict from the previous state

    def add(self, obs, prev_obs, target_value, target_policy):
        self.obs_buf[self.ptr]  = obs
        self.prev_obs_buf[self.ptr] = prev_obs
        self.target_v_buf[self.ptr] = target_value
        self.target_p_buf[self.ptr] = target_policy
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda arr: torch.from_numpy(arr[idx]).to(self.device)
        return Transition(
            to_t(self.obs_buf),
            to_t(self.prev_obs_buf),
            to_t(self.target_v_buf),
            to_t(self.target_p_buf),
        )

