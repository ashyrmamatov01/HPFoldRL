from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import random
import numpy as np
from copy import deepcopy

# -------------------------------------------------------------- #
# MCTS
# -------------------------------------------------------------- #
class Node:
    def __init__(self, env, done, parent, grid, action_index, mask, device, UCB_const):
        self.child = None
        self.parent = parent
        self.T = 0                          # total reward
        self.N = 0                          # visit count
        self.M = -float('inf')            # max value of child nodes
        self.env = env                      # environment
        self.grid = grid                    # grid
        self.done = done                    # done or not
        self.action_index = action_index    # action leading parent -> current node
        self.pred_v = 0                       # predicted value
        self.pred_p = None                    # predicted probability
        self.mask = mask                    # valid actions
        self.device = device
        self.UCB_const = UCB_const


    def getUCBscore(self):
        if self.N == 0:                     # unexplored, optimistic score
            return float('inf')

        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        value_score = (self.T / self.N)
        prior_score = self.UCB_const * self.parent.pred_p[self.action_index] * sqrt(log(top_node.N) / self.N)

        return value_score + prior_score


    def detach_parent(self):
        del self.parent                     # free memory detaching nodes
        self.parent = None


    def create_child(self):
        if self.done:                       # leaf node
            return

        child = {}
        for action in range(self.env.action_space.n):
            if self.mask[action]:       # valid action
                new_env = deepcopy(self.env)
                new_obs, _, done, _, info_dict = new_env.step(action)
                new_mask = info_dict['valid_actions']
                # env, done, parent, grid, action_index, mask, device
                child[action] = Node(new_env, done, self, new_obs, action, new_mask, self.device, self.UCB_const)

        if len(child) > 0:
            self.child = child
        else:
            assert self.done, "No valid actions found but game is not done"
            self.child = None

    def explore(self, nn_v, nn_p):
        # find a leaf node by choosing nodes with max U.
        current = self
        while current.child:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)
            action = random.Random(0).choice(actions)
            current = child[action]

        # play a random game, or expand if needed
        if current.N < 1:
            current.pred_v, current.pred_p = current.rollout(nn_v, nn_p)
            current.T = current.T + current.pred_v
            current.M = max(current.M, current.pred_v)
        else:
            current.create_child()
            if current.child:
                keys_list = list(current.child.keys())
                random_key = random.Random(0).choice(keys_list)
                current = current.child[random_key]
                
            current.pred_v, current.pred_p = current.rollout(nn_v, nn_p)
            current.T = current.T + current.pred_v
            current.M = max(current.M, current.pred_v)
        current.N += 1

        # backpropagate
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T
            parent.M = max(parent.M, current.M)


    def rollout(self, nn_v, nn_p):

        if self.done:
            return 0, None
        else:
            obs = torch.from_numpy(self.grid)
            obs = obs.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                v = nn_v(obs).cpu()
                p = nn_p(obs).cpu()

            return v.numpy().flatten()[0], p.numpy().flatten()


    def next(self):
        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')

        child = self.child

        max_N = max(node.N for node in child.values())

        prob_target = [0] * self.env.action_space.n
        for action in range(self.env.action_space.n):
            if self.mask[action]:       # valid action
                prob_target[action] = child[action].N / max_N
        prob_target /= np.sum(prob_target)

        probs = [ node.N / max_N for node in child.values() ]
        probs /= np.sum(probs)

        next_children = random.choices(list(child.values()), weights=probs)[0]
    
        return next_children, next_children.action_index, next_children.grid, prob_target, self.grid


# -------------------------------------------------------------- #
# MLP models for alphazero
# -------------------------------------------------------------- #
class MLP_Value(nn.Module):
    """MLP that maps flattened grid (B,C,H,W) → value

    Parameters
    ----------
    in_dim : int  – flattened observation size
    hidden : tuple[int,int]  – two hidden-layer sizes
    """

    def __init__(
            self, 
            in_dim: int, 
            n_actions: int, 
            hidden: tuple[int, int] = (256, 256)
        ) -> None:

        super().__init__()
        h1, h2 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # value
        self.val = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, *)
        x = x.float().view(x.size(0), -1)
        feats = self.backbone(x)
        v = self.val(feats)                       # (B,1)
        return v

class MLP_Policy(nn.Module):
    """MLP that maps flattened grid (B,C,H,W) → probability of choosing each action

    Parameters
    ----------
    in_dim : int  – flattened observation size
    n_actions : int
    hidden : tuple[int,int]  – two hidden-layer sizes
    """

    def __init__(
            self, 
            in_dim: int, 
            n_actions: int, 
            hidden: tuple[int, int] = (256, 256)
        ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # policy
        self.p_linear = nn.Linear(h2, n_actions)
        self.p_softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, *)
        x = x.float().view(x.size(0), -1)
        feats = self.backbone(x)
        p = self.p_linear(feats)                  # (B,A)
        p = self.p_softmax(p)                     # (B,A)
        return p

# -------------------------------------------------------------- #
# CNN models for alphazero
# -------------------------------------------------------------- #

class CNN_Value(nn.Module):
    """Value-network with CNN backbone  

    Args:
        board_size: int  # side length of square grid observation
        n_actions: int   # number of discrete actions
        hidden_dims: tuple[int,int] = (128, 128)  # sizes of hidden fully-connected layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (128, 128),
        ) -> None:

        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        h1, h2 = hidden_dims

        # CNN feature extractor: input channels=3 (empty/H/P one-hot)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Noisy fully-connected dueling streams
        self.fc = nn.Sequential(
            nn.Linear(64, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.val = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) int grid (0=empty,1=H,2=P)
        # one-hot encode into 3 channels
        x = x.long()
        x = F.one_hot(x, num_classes=3).permute(0, 3, 1, 2).float()
        feat = self.conv(x).view(x.size(0), -1)  # (B,64)
        v = self.val(self.fc(feat))              # (B,1)   
        return v

class CNN_Policy(nn.Module):
    """Policy network with CNN backbone 

    Args:
        board_size: int  # side length of square grid observation
        n_actions: int   # number of discrete actions
        hidden_dims: tuple[int,int] = (128, 128)  # sizes of hidden fully-connected layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (128, 128),
        ) -> None:

        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        h1, h2 = hidden_dims

        # CNN feature extractor: input channels=3 (empty/H/P one-hot)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Noisy fully-connected dueling streams
        self.fc = nn.Sequential(
            nn.Linear(64, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # policy
        self.p_linear = nn.Linear(h2, n_actions)
        self.p_softmax = nn.Softmax()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) int grid (0=empty,1=H,2=P)
        # one-hot encode into 3 channels
        x = x.long()
        x = F.one_hot(x, num_classes=3).permute(0, 3, 1, 2).float()
        feat = self.conv(x).view(x.size(0), -1)  # (B,64)
        p = self.p_linear(self.fc(feat))          # (B,A)
        p = self.p_softmax(p)                     # (B,A)`
        return p

# -------------------------------------------------------------- #
# CNN (3D) models for alphazero
# -------------------------------------------------------------- #

class CNN_Value(nn.Module):
    """Value-network with CNN backbone

    Args:
        board_size: int  # side length of square grid observation
        n_actions: int   # number of discrete actions
        hidden_dims: tuple[int,int] = (128, 128)  # sizes of hidden fully-connected layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (128, 128),
        ) -> None:

        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        h1, h2 = hidden_dims

        # CNN feature extractor: input channels=3 (empty/H/P one-hot)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Noisy fully-connected dueling streams
        self.fc = nn.Sequential(
            nn.Linear(64, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.val = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) int grid (0=empty,1=H,2=P)
        # one-hot encode into 3 channels
        x = x.long()
        x = F.one_hot(x, num_classes=3).permute(0, 3, 1, 2).float()
        feat = self.conv(x).view(x.size(0), -1)  # (B,64)
        v = self.val(self.fc(feat))              # (B,1)
        return v

class CNN_Policy(nn.Module):
    """Policy network with CNN backbone

    Args:
        board_size: int  # side length of square grid observation
        n_actions: int   # number of discrete actions
        hidden_dims: tuple[int,int] = (128, 128)  # sizes of hidden fully-connected layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (128, 128),
        ) -> None:

        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        h1, h2 = hidden_dims

        # CNN feature extractor: input channels=3 (empty/H/P one-hot)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Noisy fully-connected dueling streams
        self.fc = nn.Sequential(
            nn.Linear(64, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # policy
        self.p_linear = nn.Linear(h2, n_actions)
        self.p_softmax = nn.Softmax()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) int grid (0=empty,1=H,2=P)
        # one-hot encode into 3 channels
        x = x.long()
        x = F.one_hot(x, num_classes=3).permute(0, 3, 1, 2).float()
        feat = self.conv(x).view(x.size(0), -1)  # (B,64)
        p = self.p_linear(self.fc(feat))          # (B,A)
        p = self.p_softmax(p)                     # (B,A)`
        return p