"""Gymnasium environment for 2-D HP lattice folding."""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hp_problem.moves import Move2D
from hp_problem.utils import set_seed
from hp_problem.utils.visualize import plot_and_export, render_hp_chain_ascii

logger = logging.getLogger(__name__)


class HP2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "ascii", "rgb_array"]}
    _ABS_NEIGHBORS = [(-1,  0),
                  ( 1,  0),
                  ( 0, -1),
                  ( 0,  1)]
    
    def __init__(
        self,
        sequence: str,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        board_size: Optional[int] = None,
        reward_complete_bonus: float = 1,
        reward_step: float = 0.0,
        reward_trap_penalty_factor: float = 0.3,
        reward_illegal_move_penalty_factor: float = 0.5,
    ):
        super().__init__()
        
        # self.sequence = list(sequence.upper())
        if isinstance(sequence, str):
            self.sequence = list(sequence.upper())
        else:
            self.sequence = sequence
        assert set(self.sequence).issubset({"H", "P"}), "Sequence must be H/P only."

        self.n_res = len(self.sequence)
        self.render_mode = render_mode
        half = self.n_res // 2
        self.board_size = board_size or (2 * half + 1)
        self.offset = self.board_size // 2
        self.max_radius = self.n_res / 2.0
        self.rng = np.random.default_rng(set_seed(seed))
        self.initial_dir = np.array([1, 0], dtype=int)

        # Rewards
        self.R_COMPLETE_SUCCESS = reward_complete_bonus
        self.R_STEP = reward_step
        self.R_TRAP = -abs(float(self.n_res) * reward_trap_penalty_factor)
        self.R_ILLEGAL_MOVE = -abs(float(self.n_res) * reward_illegal_move_penalty_factor)


        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int8
        )
        self.action_space = spaces.Discrete(len(Move2D))
        self.info_str_from_last_step = "reset"
        self.reset()

    # --------------------------------------------------------------------- #
    # core helpers
    # --------------------------------------------------------------------- #
    def _valid_moves(self):
        valid = []
        for idx, vec in enumerate(Move2D.as_array()):
            nxt = self.pos + vec
            if self._in_bounds(nxt) and self.grid[tuple(nxt)] == 0:
                valid.append(idx)
        return valid

    def _in_bounds(self, xy):
        arr = np.array(xy, dtype=int)
        return np.all((0 <= arr) & (arr < self.board_size))

    def _energy(self):
        """Return −#non-adjacent H-H contacts."""
        energy = 0
        for x, y in self.H_coords:
            for dx, dy in self._ABS_NEIGHBORS:
                nx, ny = x + dx, y + dy
                if (
                    self._in_bounds((nx, ny))
                    and self.grid[nx, ny] == 1
                    and (nx, ny) not in self.backbone_adj[(x, y)]
                ):
                    energy -= 1
        return energy // 2  # each contact counted twice

    def get_coords(self):
        """Return coordinates of the backbone."""
        # coords = np.zeros((self.n_res, 2), dtype=int)
        # for i, (x, y) in enumerate(self.backbone):
        #     coords[i] = x - self.offset, y - self.offset
        # return coords
        return np.array(self.backbone, dtype=int) - self.offset
    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(set_seed(seed))
        self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.backbone_adj = {}
        self.H_coords = []

        start = np.array([self.offset, self.offset])
        self.pos = start

        res_type = self.sequence[0]
        self.grid[tuple(start)] = 1 if res_type == "H" else 2
        if res_type == "H":
            self.H_coords.append(tuple(start))
        self.backbone = [tuple(start)]
        self.step_idx = 1
        self.current_dir = self.initial_dir.copy()
        obs = self.grid.copy()
        valid_mask = self._get_valid_mask()
        self.info_str_from_last_step = "reset"
        return obs, {"valid_actions": valid_mask, "info": self.info_str_from_last_step}

    def step(self, action: int):

        # new: mask-based validity + zero reward on invalid/trapping
        current_mask = self._get_valid_mask() # Get mask for current state *before* action
        if not current_mask[action]:
            # return self.grid.copy(), 0.0, True, False, {"valid_actions": mask}
            # Agent selected an action that was already invalid from current state
            self.info_str_from_last_step = "illegal_move"
            return self.grid.copy(), self.R_ILLEGAL_MOVE, True, False, {"valid_actions": current_mask, "info": self.info_str_from_last_step}


        relative_move_enum_val = Move2D.as_array()[action] # This is like (-1,0), (0,1), or (1,0)
            
        # self.current_dir is the current "forward" direction of the agent
        # relative_move_enum_val[0] is change along agent's right axis (-1 Left, 0 Fwd, 1 Right)
        # relative_move_enum_val[1] is change along agent's forward axis (should be 1 for FWD, 0 for turns)

        # Determine the absolute grid displacement vector
        # _to_absolute_move correctly calculates this based on current_dir and relative_move_enum_val
        absolute_move_vec = self._to_absolute_move(relative_move_enum_val) # Grid displacement
        nxt = self.pos + absolute_move_vec

        # Update current_dir IF the action was a turn.
        # A turn means the agent's new "forward" is the direction it just moved.
        # If the action was "FORWARD" (relative_move_enum_val = (0,1)), current_dir doesn't change its orientation.
        # Forward is (0,1), Left is (-1,0), Right is (1,0) in Move2D.value
        
        # Check if the action corresponds to turning left or right.
        # Move2D.FORWARD.value is (0,1). If relative_move_enum_val is not (0,1), it's a turn.
        # If LEFT or RIGHT, current_dir *becomes* the absolute_move_vec because that's the new forward.
        if not np.array_equal(relative_move_enum_val, Move2D.FORWARD.value):
            self.current_dir = absolute_move_vec # New orientation is the direction of the turn


        self.backbone_adj[tuple(self.pos)] = (
            self.backbone_adj.get(tuple(self.pos), set()) | {tuple(nxt)}
        )
        self.backbone_adj[tuple(nxt)] = (
            self.backbone_adj.get(tuple(nxt), set()) | {tuple(self.pos)}
        )
        self.pos = nxt

        res_type = self.sequence[self.step_idx]
        self.grid[tuple(nxt)] = 1 if res_type == "H" else 2
        if res_type == "H":
            self.H_coords.append(tuple(nxt))
        self.backbone.append(tuple(nxt))
        self.step_idx += 1

        terminated_by_completion = self.step_idx == self.n_res
        next_obs_grid = self.grid.copy()
        actual_energy = self._energy() # this is negative energy, so lower is better

        if terminated_by_completion:
            # reward = self.R_COMPLETE_SUCCESS
            # reward = float(-1 * actual_energy)
            reward = self.R_COMPLETE_SUCCESS + (-1 * actual_energy)
            terminated = True
            self.info_str_from_last_step = "completed"
            # return self.grid.copy(), reward, True, False, {"valid_actions": current_mask, "info": info_str} # current_mask was for the state *before* this terminal action
            mask_at_s_prime = current_mask
        else:
            # mask_at_s_prime is the valid mask for the NEW state (next_obs_grid, self.pos)
            mask_at_s_prime = self._get_valid_mask()
            # Not completed yet, so check if the next state is a trap
            if not mask_at_s_prime.any():
                # No valid moves from the next state, so it's a trap
                reward = self.R_TRAP
                # reward = float(-1 * actual_energy)
                # reward = self.R_TRAP + float(-1 * actual_energy)
                self.info_str_from_last_step = "trapped"
                terminated = True
            else:
                # Still valid moves, so not a trap
                reward = self.R_STEP
                self.info_str_from_last_step = "step_ok"
                terminated = False

        return next_obs_grid, reward, terminated, False, {"valid_actions": mask_at_s_prime, "info": self.info_str_from_last_step}
        # if terminated:
        #     reward = float(-1 * actual_energy)
        #     return self.grid.copy(), reward, True, False, {"valid_actions": current_mask}

        # next_mask = self._get_valid_mask()
        # Check if trapped after this move (no valid next moves)
        # This makes the current move lead to a terminal state if it's a trap
        # This is important because if next_mask has no True values, the agent might still
        # try to take an action, and the environment's initial check in step() will catch it.
        # However, for learning, it's better if `done` is True when truly trapped.
        # if not terminated and not next_mask.any():
            # terminated = True # Trapped
            # Keep the reward as 0 unless it was the final step already.
            # Or assign a penalty for trapping if desired: reward = -penalty_for_trap
        # return self.grid.copy(), reward, terminated, False, {"valid_actions": next_mask}

        
    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(
        self,
        mode: Optional[str] = None,
        close: bool = False,
        filename: Optional[str] = None,
        **plot_kwargs,
    ):
        if close:
            return
        if mode is None:
            mode = self.render_mode

        if mode == "human":
            plot_and_export(self.sequence, self.backbone, filename, **plot_kwargs)
        elif mode == "ascii":
            # print(self.grid)
            _raw_txt = render_hp_chain_ascii(self.sequence, self.backbone, fill_char='.', pad=1, **plot_kwargs)
            return _raw_txt
        elif mode == "rgb_array":
            return np.array(self.backbone, dtype=int)

    def _to_absolute_move(self, rel: np.ndarray) -> np.ndarray:
        """Convert a relative move into absolute grid shift based on current direction."""
        fwd = self.current_dir
        right = np.array([fwd[1], -fwd[0]], dtype=int)
        return rel[0] * right + rel[1] * fwd

    def _get_valid_mask(self) -> np.ndarray:
        """Mask invalid, out-of-bounds, collision, or trap moves."""
        mask = np.zeros(len(Move2D), dtype=bool)
        remaining = self.n_res - self.step_idx
        for idx, rel in enumerate(Move2D.as_array()):
            vec = self._to_absolute_move(rel)
            nxt = self.pos + vec
            # out of grid
            if not self._in_bounds(nxt):
                continue
            # overlap
            if self.grid[tuple(nxt)] != 0:
                continue
            # bounding region
            coord = nxt - self.offset
            if np.any(np.abs(coord) > self.max_radius):
                continue
            # trap-check via DFS on remaining placements
            temp = self.grid.copy()
            temp[tuple(nxt)] = 1 if self.sequence[self.step_idx] == "H" else 2
            if remaining > 0 and not self._connectivity_feasible(temp, tuple(nxt), remaining):
                continue
            mask[idx] = True
        return mask

    def _connectivity_feasible(
        self, grid: np.ndarray, pos: Tuple[int, int], rem: int
    ) -> bool:
        """Ensure at least `rem` empty cells reachable from `pos`."""
        visited = {pos}
        stack = [pos]
        count = 0
        while stack and count < rem:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (x + dx, y + dy)
                if (
                    0 <= nb[0] < self.board_size
                    and 0 <= nb[1] < self.board_size
                    and grid[nb] == 0
                    and nb not in visited
                ):
                    visited.add(nb)
                    stack.append(nb)
                    count += 1
        return count >= rem
