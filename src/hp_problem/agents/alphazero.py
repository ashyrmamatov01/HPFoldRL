from __future__ import annotations
from copy import deepcopy
import logging, math, numpy as np, torch, torch.nn.functional as F

from hp_problem.models.alphazero_models import Node
from hp_problem.utils.replay_buffer_policy import ReplayBuffer

#MCTS_POLICY_EXPLORE = 200

class AlphaZeroAgent:
    """AlphaZero agent for MCTS with neural network."""

    def __init__(
        self, 
        ############## Environment ###############
        obs_shape: tuple[int, ...],
        n_actions: int,
        total_steps: int,
        env: object,

        ############## Networks ###############
        network_type: str = "mlp",
        board_size: int | None = None,            # for CNN
        hidden: tuple[int,int] = (256,256),       # for MLP
        cnn_hidden: tuple[int,int] = (128,128),   # for CNN
        device: str = "cuda" if torch.cuda.is_available() else "cpu",

        ############## Hyperparameters #############
        gamma: float = 0.99,
        lr_v: float = 2e-4,                     # learning rate for value network
        lr_p: float = 2e-4,                     # learning rate for policy network   
        batch_size: int = 64,
        buffer_size: int = 100_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 0.8,
        seed: int | None = None,

        ############## MCTS ############### 
        
        UCB_const=1.0,
        MCTS_simulation_count=200,

        ) -> None:

        self.device = device
        in_dim = int(np.prod(obs_shape))

        if network_type == "mlp":
            from hp_problem.models.alphazero_models import MLP_Value, MLP_Policy
            self.nn_v = MLP_Value(in_dim, n_actions, hidden).to(device)
            self.nn_p = MLP_Policy(in_dim, n_actions, hidden).to(device)
        elif network_type == "cnn":
            from hp_problem.models.alphazero_models import CNN_Value, CNN_Policy
            self.nn_v = CNN_Value(board_size, n_actions, cnn_hidden).to(device)
            self.nn_p = CNN_Policy(board_size, n_actions, cnn_hidden).to(device)
        elif network_type == "attn":
            from hp_problem.models.attention_q_network import Attn_Value, Attn_Policy
            self.nn_v = Attn_Value(board_size, n_actions).to(device)
            self.nn_p = Attn_Policy(board_size, n_actions).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
            
        self.optim_v = torch.optim.Adam(self.nn_v.parameters(), lr=lr_v)
        self.optim_p = torch.optim.Adam(self.nn_p.parameters(), lr=lr_p)

        self.replay = ReplayBuffer(buffer_size, obs_shape, n_actions, device)
        self.gamma, self.batch_size = gamma, batch_size
        self.n_actions = n_actions

        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = int(total_steps * eps_decay_steps)
        self.global_step, self.learn_step = 0, 0
        self.rng = np.random.default_rng(seed)

        self.UCB_const = UCB_const
        self.MCTS_simulation_count = MCTS_simulation_count
        self.env = env

    def reset(self, seed: int | None = None):
        """Reset the agent for a new episode."""

        # Reset the environment and the root node
        obs, info = self.env.reset(seed=seed)
        mask = info["valid_actions"]
        env_cpy = deepcopy(self.env)

        # Create a new root node
        # env, done, parent, grid
        # action_index: forward
        # mask: all the actions are possible for the initial step
        self.tree = Node(env_cpy, False, 0, obs, 1, mask, self.device, self.UCB_const) 
        return obs, info
    
    def store(self, *args):
        self.replay.add(*args)

    def simulate_MCTS(self):
        """Simulate MCTS for a given number of iterations."""
        # Simulate MCTS for a given number of iterations
        for i in range(self.MCTS_simulation_count):
            self.tree.explore(self.nn_v, self.nn_p)

        # Get the next tree, action, observation, policy and policy observation
        next_tree, next_action, obs, p, prev_obs = self.tree.next()

        # Update the root node
        next_tree.detach_parent()
        self.tree = next_tree

        return next_tree, next_action, obs, p, prev_obs
    
    def _learn_v(self, samples):
        # Compute the value loss: Mean Squared Error between predicted and actual values
        pred_v = self.nn_v(samples.obs)
        target = samples.target_v
        target = torch.unsqueeze(target, 1)
        loss_v = F.mse_loss(pred_v, target)
        self.optim_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_v.parameters(), 10.0)
        self.optim_v.step()
        return loss_v.item()

    def _learn_p(self, samples):
        # Compute the policy loss: Categorical Crossentropy between predicted and actual policy
        pred_p = self.nn_p(samples.prev_obs)
        target_p = samples.target_p
        loss_p = F.cross_entropy(pred_p, target_p)
        self.optim_p.zero_grad()
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_p.parameters(), 10.0)
        self.optim_p.step()
        return loss_p.item()
    
    def learn(self):
        """Train the neural networks using the replay buffer."""
        if self.replay.size < self.batch_size:
            return None

        # Sample a batch of experiences from the replay buffer
        # "obs", "prev_obs", "target_value", "target_policy"
        samples = self.replay.sample(self.batch_size)
        # Compute the value loss
        loss_v = self._learn_v(samples)
        # Compute the policy loss
        loss_p = self._learn_p(samples)
        return [loss_v, loss_p]

