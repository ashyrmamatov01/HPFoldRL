"""Gymnasium environment for 3-D HP folding (simple cubic lattice)."""
from __future__ import annotations
import logging
from typing import Optional, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from hp_problem.moves import Move3D
from hp_problem.utils import set_seed
from hp_problem.utils.visualize import  plot_and_export

logger = logging.getLogger(__name__)


class HP3DEnv(gym.Env):
    """Coordinate-based 3D HP folding env with symmetry-breaking and trap detection."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, sequence: str, seed: Optional[int] = None):
        super().__init__()
        assert set(sequence).issubset({"H", "P"}), "Sequence must be H/P string"
        self.sequence = sequence.upper()
        self.length = len(self.sequence)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Action space: 0=forward,1=up,2=down,3=left,4=right
        self.action_space = spaces.Discrete(5)

        # Define bounding region based on sequence length
        self.radius = self.length // 2

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-float(self.radius),
            high=float(self.radius),
            shape=(self.length * 5,),
            dtype=np.float32
        )
        
        # Symmetry-breaking flags
        self.has_non_forward_turn = False
        self.has_z_deviation = False


        self.reset()

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Initialize first two residues and flags."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Place first two residues along +X
        self.positions: List[Tuple[int, int, int]] = [(0, 0, 0), (1, 0, 0)]
        self.current_index = 2
        self.done = False
        self.has_non_forward_turn = False
        self.has_z_deviation = False
        return self._get_observation(), {}

    def step(self, action: int):
        """Place next residue; return obs, reward, terminated, truncated, info."""
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        # If all placed, finalize
        if self.current_index >= self.length:
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Compute next position
        next_pos = self._get_next_position(action)
        # Invalid = out-of-bounds or collision
        if not self._in_bounds(next_pos) or next_pos in self.positions:
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, False, {}
            return self._get_observation(), 0.0, False, False, {}

        # Valid move: place residue
        self.positions.append(next_pos)
        # Update symmetry flags
        if not self.has_non_forward_turn and action != 0:
            if action == 4:
                self.has_non_forward_turn = True
        if not self.has_z_deviation and action == 1:
            self.has_z_deviation = True

        self.current_index += 1
        # Check termination
        if self.current_index == self.length:
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, False, {}
        if self._is_trapped():
            self.done = True
            return self._get_observation(), 0.0, True, False, {}

        return self._get_observation(), 0.0, False, False, {}

    def render(self, mode="human", savepath=None, **plot_kwargs):
        """
        mode: 'human' or 'rgb_array'
        savepath: optional path to save the plot
        plot_kwargs: passed to plot_and_export (colors, title, etc.)
        """
        seq = self.sequence
        coords = self.coords  # current folding coords array
        # choose 2D or 3D based on env
        img = plot_and_export(seq, coords,
                              three_d=self.is_3d,
                              mode=mode,
                              savepath=savepath,
                              **plot_kwargs)
        return img

    def close(self):
        # override if needed to close plots
        pass


    def _get_observation(self) -> np.ndarray:
        """Return flattened obs of shape (length * 5,): x,y,z,type,index_norm or -1 for unplaced."""
        obs = -np.ones((self.length, 5), dtype=np.float32)
        denom = (self.length - 1) or 1
        for i, pos in enumerate(self.positions):
            obs[i, 0:3] = pos
            obs[i, 3] = 1.0 if self.sequence[i] == "H" else 0.0
            obs[i, 4] = i / denom
        return obs.flatten()

    def _get_direction_vector(self) -> Tuple[int, int, int]:
        """Return vector from second-last to last placed residue."""
        if len(self.positions) < 2:
            return (1, 0, 0)
        p1, p2 = self.positions[-2], self.positions[-1]
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    def _get_next_position(self, action: int) -> Tuple[int, int, int]:
        """Map a relative action to an absolute 3D coordinate."""
        f = self._get_direction_vector()
        # Define local axes
        if f in [(1, 0, 0), (-1, 0, 0)]:
            up, down = (0, 0, 1), (0, 0, -1)
            left = (0, 1, 0) if f == (1, 0, 0) else (0, -1, 0)
            right = (0, -1, 0) if f == (1, 0, 0) else (0, 1, 0)
        elif f in [(0, 1, 0), (0, -1, 0)]:
            up, down = (0, 0, 1), (0, 0, -1)
            left = (1, 0, 0) if f == (0, 1, 0) else (-1, 0, 0)
            right = (-1, 0, 0) if f == (0, 1, 0) else (1, 0, 0)
        else:
            up, down = (0, 1, 0), (0, -1, 0)
            left, right = (1, 0, 0), (-1, 0, 0)
        moves = {0: f, 1: up, 2: down, 3: left, 4: right}
        x, y, z = self.positions[-1]
        dx, dy, dz = moves.get(action, f)
        return (x + dx, y + dy, z + dz)

    def _in_bounds(self, pos: Tuple[int, int, int]) -> bool:
        """Check if position is within +/- radius = length//2."""
        r = self.length // 2
        return all(abs(c) <= r for c in pos)

    def _is_trapped(self) -> bool:
        """Return True if no valid next placement exists."""
        for a in range(self.action_space.n):
            np_ = self._get_next_position(a)
            if self._in_bounds(np_) and np_ not in self.positions:
                return False
        return True

    def _calculate_hh_bonds(self) -> float:
        """Count non-sequential H–H adjacent contacts as reward."""
        coords_h = [self.positions[i] for i in range(self.length) if self.sequence[i] == "H"]
        count = 0
        for i in range(len(coords_h)):
            for j in range(i + 1, len(coords_h)):
                if sum(abs(coords_h[i][k] - coords_h[j][k]) for k in range(3)) == 1:
                    idx_i = self.positions.index(coords_h[i])
                    idx_j = self.positions.index(coords_h[j])
                    if abs(idx_i - idx_j) > 1:
                        count += 1
        return float(count)