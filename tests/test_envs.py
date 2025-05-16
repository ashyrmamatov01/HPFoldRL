"""Simple pytest smoke-tests for HP environments."""
from hp_problem.envs import HP2DEnv, HP3DEnv


def _run(env_cls):
    env = env_cls("HPHPPH")
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
    assert obs is not None


def test_hp2d():
    _run(HP2DEnv)


def test_hp3d():
    _run(HP3DEnv)
