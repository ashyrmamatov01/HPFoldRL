from __future__ import annotations
import random, collections, numpy as np, torch

Transition = collections.namedtuple(
    "Transition", ("obs", "action", "reward", "next_obs", "done", "mask")
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
        self.next_buf = np.zeros_like(self.obs_buf)
        self.act_buf  = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf  = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.bool_)
        self.mask_buf = np.zeros((capacity, n_actions), dtype=np.bool_)  # valid-action mask

    def add(self, obs, action, reward, next_obs, done, mask):
        self.obs_buf[self.ptr]  = obs
        self.next_buf[self.ptr] = next_obs
        self.act_buf[self.ptr]  = action
        self.rew_buf[self.ptr]  = reward
        self.done_buf[self.ptr] = done
        self.mask_buf[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda arr: torch.from_numpy(arr[idx]).to(self.device)
        return Transition(
            to_t(self.obs_buf),
            to_t(self.act_buf),
            to_t(self.rew_buf),
            to_t(self.next_buf),
            to_t(self.done_buf),
            to_t(self.mask_buf),
        )
