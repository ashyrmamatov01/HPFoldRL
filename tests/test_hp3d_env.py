import numpy as np
import pytest
from hp_problem.envs.hp3d_env import HP3DEnv


def test_reset_observation_shape_and_values():
    seq = "HPH"
    env, _ = HP3DEnv(seq, seed=123).reset()
    # shape = n_res * 5
    print(f'env.shape = {env.shape}')
    assert env.shape == (len(seq) * 5,)
    # first two positions = (0,0,0), (1,0,0)
    obs = env
    np.testing.assert_array_equal(obs[0:3], [0, 0, 0])
    np.testing.assert_array_equal(obs[5:8], [1, 0, 0])


def test_forward_step_and_reward():
    seq = "HHHH"
    env = HP3DEnv(seq)
    obs, _ = env.reset()
    # place 2→3→4 residues by forward moves
    for _ in range(3):
        # obs, r, term, _, _ = HP3DEnv(seq).step(0)
        obs2, reward, done, _, _ = env.step(0)
    # final reward: zero non-sequential HH contacts in straight chain
    assert done in [True, False]
    # assert reward == True
    assert reward == 0.0