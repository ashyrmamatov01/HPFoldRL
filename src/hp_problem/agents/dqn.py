from __future__ import annotations
import logging, math, numpy as np, torch, torch.nn.functional as F
from hp_problem.models.q_network import DuelingQNet, CNNDuelingQNet
from hp_problem.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class DQNAgent:
    """Double-Dueling DQN with target network & ε-decay identical to TabularQAgent."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        total_steps: int,
        *,
        network_type: str = "mlp",
        board_size: int | None = None,            # for CNN
        hidden: tuple[int,int] = (256,256),       # for MLP
        cnn_hidden: tuple[int,int] = (128,128),   # for CNN

        gamma: float = 0.99,
        lr: float = 2e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        update_target_every: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 0.8,
        seed: int | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        in_dim = int(np.prod(obs_shape))
        # self.online = DuelingQNet(in_dim, n_actions).to(device)
        # self.target = DuelingQNet(in_dim, n_actions).to(device)
        if network_type == "mlp":
            in_dim = int(np.prod(obs_shape))
            self.online = DuelingQNet(in_dim, n_actions, hidden).to(device)
            self.target = DuelingQNet(in_dim, n_actions, hidden).to(device)
        elif network_type == "cnn":
            assert board_size is not None, "board_size required for CNN"
            self.online = CNNDuelingQNet(board_size, n_actions, cnn_hidden).to(device)
            self.target = CNNDuelingQNet(board_size, n_actions, cnn_hidden).to(device)
        elif network_type == "attn":
            assert board_size is not None, "board_size required for attn network"
            from hp_problem.models.attention_q_network import AttnDuelingQNet
            self.online = AttnDuelingQNet(board_size, n_actions).to(device)
            self.target = AttnDuelingQNet(board_size, n_actions).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
        self.target.load_state_dict(self.online.state_dict())

        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_size, obs_shape, n_actions, device)
        self.gamma, self.batch_size = gamma, batch_size
        self.update_target_every = update_target_every
        self.n_actions = n_actions

        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = int(total_steps * eps_decay_steps)
        self.global_step, self.learn_step = 0, 0
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------- #
    def _epsilon(self) -> float:
        if self.global_step >= self.eps_decay_steps:
            return self.eps_end
        frac = self.global_step / self.eps_decay_steps
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray) -> int:
        eps = self._epsilon()
        self.global_step += 1
        if self.rng.random() < eps:
            return int(self.rng.choice(np.flatnonzero(valid_mask)))
        with torch.no_grad():
            q = self.online(torch.from_numpy(obs[None]).to(self.device))
            q[~torch.from_numpy(valid_mask[None]).to(self.device)] = -1e9
            return int(q.argmax(dim=1).item())

    # -------------------------------------------------------------- #
    def store(self, *args):
        self.replay.add(*args)

    def _compute_td_loss(self):
        if self.replay.size < self.batch_size:
            return None
        trans = self.replay.sample(self.batch_size)
        q = self.online(trans.obs).gather(1, trans.action.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_online = self.online(trans.next_obs)
            next_q_online[~trans.mask] = -1e9
            next_actions = next_q_online.argmax(1, keepdim=True)
            next_q_target = self.target(trans.next_obs).gather(1, next_actions).squeeze(1)
            target = trans.reward + self.gamma * (1 - trans.done.float()) * next_q_target
        loss = F.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()
        return loss.item()

    def learn(self):
        loss = self._compute_td_loss()
        if loss is not None:
            self.learn_step += 1
            if self.learn_step % self.update_target_every == 0:
                self.target.load_state_dict(self.online.state_dict())
        return loss
