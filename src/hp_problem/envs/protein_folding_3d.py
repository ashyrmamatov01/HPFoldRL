import gymnasium as gym
import torch
import numpy as np
import random
import logging
import os
import matplotlib.pyplot as plt
import time
import csv

from datetime import datetime


class HPProteinFoldingEnv(gym.Env):
    """
    Coordinate-based environment with 3D rendering capability.
    Includes trap detection, symmetry-breaking constraints, and
    a bounded coordinate system limiting positions to within +/- (length/2).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, sequence="PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP"):
        super(HPProteinFoldingEnv, self).__init__()
        self.sequence = sequence
        self.length = len(sequence)
        self.action_space = gym.spaces.Discrete(5)

        # Define bounding region based on sequence length
        self.radius = self.length // 2

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-float(self.radius),
            high=float(self.radius),
            shape=(self.length * 5,),
            dtype=np.float32
        )

        # Flags to handle symmetry-breaking
        self.has_non_forward_turn = False
        self.has_z_deviation = False

        self.reset()

    def reset(self):
        self.positions = [(0, 0, 0), (1, 0, 0)]
        self.current_index = 2
        self.done = False
        self.has_non_forward_turn = False
        self.has_z_deviation = False
        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, {}

        if self.current_index >= self.length:
            # All placed
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, {}

        next_pos = self._get_next_position(action)
        # Check if next_pos is out of bound
        if not self._in_bounds(next_pos):
            # Treat out-of-bounds as invalid
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

        if next_pos in self.positions:
            # Invalid action, no move
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

        # Valid move
        self.positions.append(next_pos)
        self.current_index += 1

        # Update the symmetry-breaking flags after a valid move
        if not self.has_non_forward_turn and action != 0:
            # The first non-forward action must be to the right (action=4)
            if action == 4:
                self.has_non_forward_turn = True

        if not self.has_z_deviation and action in {1, 2}:
            # The first vertical deviation must be up (action=1).
            if action == 1:
                self.has_z_deviation = True

        if self.current_index == self.length:
            # Finished
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, {}
        else:
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

    def render(self, mode='human', show_dialog=True, filename=None):
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, 'k-', lw=2)

        for i, (x, y, z) in enumerate(self.positions):
            c = 'r' if self.sequence[i] == 'H' else 'b'
            ax.scatter(x, y, z, c=c, s=100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if filename:
            plt.savefig(filename)
        if show_dialog:
            plt.show()
        else:
            plt.close()

    def close(self):
        plt.close('all')

    def _get_observation(self):
        obs = np.zeros((self.length, 5), dtype=np.float32)
        denom = (self.length - 1) if self.length > 1 else 1
        for i, pos in enumerate(self.positions):
            x, y, z = pos
            aatype = 1 if self.sequence[i] == 'H' else 0
            index_norm = i / denom
            obs[i] = [x, y, z, aatype, index_norm]
        for i in range(self.current_index, self.length):
            obs[i] = [0, 0, 0, -1, -1]
        return obs.flatten()

    def _get_direction_vector(self):
        if len(self.positions) < 2:
            return (1, 0, 0)
        p1 = self.positions[-2]
        p2 = self.positions[-1]
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    def _get_next_position(self, action):
        f = self._get_direction_vector()
        if f == (1, 0, 0) or f == (-1, 0, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (0, 1, 0) if f == (1, 0, 0) else (0, -1, 0)
            right = (0, -1, 0) if f == (1, 0, 0) else (0, 1, 0)
        elif f == (0, 1, 0) or f == (0, -1, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (1, 0, 0) if f == (0, 1, 0) else (-1, 0, 0)
            right = (-1, 0, 0) if f == (0, 1, 0) else (1, 0, 0)
        else:
            # f=(0,0,1) or f=(0,0,-1)
            up = (0, 1, 0) if f == (0, 0, 1) else (0, -1, 0)
            down = (0, -1, 0) if f == (0, 0, 1) else (0, 1, 0)
            left = (1, 0, 0)
            right = (-1, 0, 0)

        current_pos = self.positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0] + move[0], current_pos[1] + move[1], current_pos[2] + move[2])

    def _calculate_hh_bonds(self):
        coords_h = []
        for i, pos in enumerate(self.positions):
            if self.sequence[i] == 'H':
                coords_h.append((i, pos))
        count = 0
        n = len(coords_h)
        for i in range(n):
            for j in range(i + 1, n):
                idx_i, p_i = coords_h[i]
                idx_j, p_j = coords_h[j]
                if abs(idx_i - idx_j) > 1:
                    if self._are_adjacent(p_i, p_j):
                        count += 1
        return count

    def _are_adjacent(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        dz = abs(p1[2] - p2[2])
        return (dx + dy + dz) == 1

    def _in_bounds(self, pos):
        x, y, z = pos
        return (abs(x) <= self.radius and abs(y) <= self.radius and abs(z) <= self.radius)

    def _is_trapped(self):
        if self.current_index < self.length:
            for a in range(self.action_space.n):
                next_pos = self._get_next_position(a)
                # Check out-of-bounds
                if not self._in_bounds(next_pos):
                    continue
                if next_pos not in self.positions:
                    return False
            return True
        return False
    
    def can_finish_dfs(self, positions, current_index):
        """
        DFS to check if we can eventually place all residues from the current state.
        Also skip any action leading out of the bounding region.
        """
        if current_index >= self.length:
            return True

        all_invalid = True
        for a in range(self.action_space.n):
            next_pos = self._get_next_position_dfs(positions, a)
            # Check out-of-bounds
            if not self._in_bounds(next_pos):
                continue
            if next_pos not in positions:
                all_invalid = False
                new_positions = positions + [next_pos]
                if self.can_finish_dfs(new_positions, current_index+1):
                    return True
        if all_invalid:
            return False
        return True

    def _get_next_position_dfs(self, positions, action):
        if len(positions) < 2:
            f = (1,0,0)
        else:
            f = (positions[-1][0]-positions[-2][0],
                 positions[-1][1]-positions[-2][1],
                 positions[-1][2]-positions[-2][2])

        if f == (1,0,0) or f == (-1,0,0):
            up = (0,0,1)
            down = (0,0,-1)
            left = (0,1,0) if f==(1,0,0) else (0,-1,0)
            right = (0,-1,0) if f==(1,0,0) else (0,1,0)
        elif f == (0,1,0) or f == (0,-1,0):
            up = (0,0,1)
            down = (0,0,-1)
            left = (1,0,0) if f==(0,1,0) else (-1,0,0)
            right = (-1,0,0) if f==(0,1,0) else (1,0,0)
        else:
            up = (0,1,0) if f==(0,0,1) else (0,-1,0)
            down = (0,-1,0) if f==(0,0,1) else (0,1,0)
            left = (1,0,0)
            right = (-1,0,0)

        current_pos = positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0]+move[0], current_pos[1]+move[1], current_pos[2]+move[2])

    def get_valid_actions_dfs(self):
        valid_mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            # Symmetry-breaking constraints
            if not self.has_non_forward_turn and a != 0 and a != 4:
                continue
            if not self.has_z_deviation and a in {1,2} and a != 1:
                continue

            next_pos = self._get_next_position(a)
            # Check bounding region
            if not self._in_bounds(next_pos):
                continue
            if next_pos in self.positions:
                continue
            new_positions = self.positions + [next_pos]
            current_idx = self.current_index + 1
            if self.can_finish_dfs(new_positions, current_idx):
                valid_mask[a] = True

        return valid_mask


    def get_valid_actions(self):
        """
        Simple version without dfs checking.
        This is a simplified version without DFS, which can save some computational complexity,
        But can still prevent most of the trapping senarios by checking one more step.
        """
        valid_mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            # Symmetry-breaking constraints
            if not self.has_non_forward_turn and a != 0 and a != 4:
                continue
            if not self.has_z_deviation and a in {1, 2} and a != 1:
                continue

            next_pos = self._get_next_position(a)
            # Check bounding region
            if not self._in_bounds(next_pos):
                continue
            if next_pos in self.positions:
                continue

            # Check if the agent is not trapped after taking this action
            # Simplified: Do not perform DFS, just ensure that at least one valid move remains after this action
            temp_positions = self.positions + [next_pos]
            temp_current_index = self.current_index + 1
            if not self._is_trapped_after_action(temp_positions, temp_current_index):
                valid_mask[a] = True

        return valid_mask

    def _is_trapped_after_action(self, positions, current_index):
        """
        Check if after taking an action, the agent is not immediately trapped.
        """
        if current_index >= self.length:
            return False  # No further actions needed

        for a in range(self.action_space.n):
            next_pos = self._get_next_position_after_action(positions, a)
            if not self._in_bounds(next_pos):
                continue
            if next_pos in positions:
                continue
            # If at least one valid move exists, not trapped
            return False
        return True

    def _get_next_position_after_action(self, positions, action):

        if len(positions) < 2:
            f = (1, 0, 0)
        else:
            f = (positions[-1][0] - positions[-2][0],
                 positions[-1][1] - positions[-2][1],
                 positions[-1][2] - positions[-2][2])

        if f == (1, 0, 0) or f == (-1, 0, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (0, 1, 0) if f == (1, 0, 0) else (0, -1, 0)
            right = (0, -1, 0) if f == (1, 0, 0) else (0, 1, 0)
        elif f == (0, 1, 0) or f == (0, -1, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (1, 0, 0) if f == (0, 1, 0) else (-1, 0, 0)
            right = (-1, 0, 0) if f == (0, 1, 0) else (1, 0, 0)
        else:
            # f=(0,0,1) or f=(0,0,-1)
            up = (0, 1, 0) if f == (0, 0, 1) else (0, -1, 0)
            down = (0, -1, 0) if f == (0, 0, 1) else (0, 1, 0)
            left = (1, 0, 0)
            right = (-1, 0, 0)

        current_pos = positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0] + move[0], current_pos[1] + move[1], current_pos[2] + move[2])
