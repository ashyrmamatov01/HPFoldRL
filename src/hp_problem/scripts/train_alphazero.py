#!/usr/bin/env python
"""Train a DQN agent on HP2DEnv with robust logging, CSV output, and graceful handling."""
from __future__ import annotations
import argparse, datetime, json, logging, pathlib, pickle, csv, signal, sys
import numpy as np
import torch
from tqdm.auto import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from hp_problem.envs.hp2d_env import HP2DEnv
from hp_problem.agents.alphazero import AlphaZeroAgent
from hp_problem.utils.visualize import _plot_metrics
from hp_problem.utils import extract_params

# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
def save_and_exit(agent, episode_log: list[dict], out_dir: pathlib.Path) -> None:
    """Save intermediate artifacts on interrupt and exit."""
    try:
        torch.save(agent.nn_v.state_dict(), out_dir / "alphazero_v_weights_interrupt.pt")
        torch.save(agent.nn_p.state_dict(), out_dir / "alphazero_p_weights_interrupt.pt")
        pickle.dump(episode_log, (out_dir / "replay_interrupt.pkl").open("wb"))
        logger.info("Saved interrupt artifacts.")
    except Exception as e:
        logger.error(f"Error saving interrupt artifacts: {e}")
    sys.exit(0)

def main(args) -> None:

    # setup output dirs
    out_dir = pathlib.Path(args.outdir).expanduser() / args.network_type / args.sequence / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # initialize env & agent
    env = HP2DEnv(sequence=args.sequence, render_mode="ascii")
    total_steps = args.episodes * env.n_res

    device = torch.device(args.device)

    agent = AlphaZeroAgent(
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        total_steps=total_steps,
        env=env,

        network_type=args.network_type,
        board_size=env.board_size,
        hidden=tuple(args.hidden_dims),
        cnn_hidden=tuple(args.cnn_hidden),

        gamma=args.gamma,
        lr_v=args.lr_v,
        lr_p=args.lr_p,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,

        UCB_const=args.UCB_const,
        MCTS_simulation_count=args.MCTS_simulation_count,
    )

    # save config
    (out_dir / "args.json").write_text(
        json.dumps({"args": vars(args), "env": extract_params(env)}, indent=2)
    )
    print(open(out_dir / "args.json").read())

    # prepare CSV logging
    csv_path = out_dir / "training_log.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["episode", "Reward", "Energy", "Actions"])
    csv_writer.writeheader()

    # handle SIGINT for graceful exit
    episode_log: list[dict] = []
    signal.signal(signal.SIGINT, lambda *_: save_and_exit(agent, episode_log, out_dir))

    rng = np.random.default_rng(args.seed)
    best_E = float("inf")

    # training loop with tqdm
    with trange(1, args.episodes + 1, desc="Episodes") as pbar:
        for ep in pbar:

            # reset the environment
            ob, info_dict = agent.reset(seed=int(rng.integers(1e9)))
            done, ep_reward = False, 0.0
        
            obs = []
            ps = []
            prev_obs = []
            masks = []
            prev_masks = []
            info = {}

            last_loss_v = 0
            last_loss_p = 0

            # run MCTS simulation
            actions = []
            while not done:
                prev_mask = agent.tree.mask
                new_tree, next_action, ob, p, prev_ob = agent.simulate_MCTS()
                mask = new_tree.mask
                _, reward, done, _, info_dict = agent.env.step(next_action) 
                ep_reward += reward

                obs.append(ob)
                ps.append(p)
                prev_obs.append(prev_ob)
                masks.append(mask)
                prev_masks.append(prev_mask)
                info = info_dict
                actions.append(next_action)

                # Store the experience in the replay buffer
                # obs, prev_obs, mask, prev_mask, target_value, target_policy
                if done:
                    for i in range(len(obs)):
                        agent.store(obs[i], prev_obs[i],
                                    ep_reward, ps[i])
                    agent.env.close()
                    break

            # compute metrics
            phys_E = agent.env._energy()
            # save best ASCII snapshot
            if phys_E < best_E and info.get("info") == "completed":
                best_E = phys_E
                best_ascii = env.render(mode="ascii", legend=True)
                (out_dir / "best_ascii.txt").write_text(env.render(mode="ascii", legend=True))

            if (ep % args.update_interval == 0):
                # learn from the replay buffer
                losses = agent.learn()

                if losses is not None:
                    loss_v, loss_p = losses
                    # tensorboard logging
                    writer.add_scalar("Train/EpReward", ep_reward, ep)
                    if loss_v is not None and np.isfinite(loss_v):
                        writer.add_scalar("Train/LossV", loss_v, ep)
                        last_loss_v = loss_v
                    if loss_p is not None and np.isfinite(loss_p):
                        writer.add_scalar("Train/LossP", loss_p, ep)
                        last_loss_p = loss_p

            # periodic console log
            if ep % args.log_interval == 0:
                current_ascii = env.render(mode="ascii", legend=True)
                avg_reward = np.mean([log["Reward"] for log in episode_log[-args.log_interval:]])
                tqdm.write(
                    f"{'-' * 20}\n"
                    f"Current ASCII:\n"
                    f"{current_ascii}"
                    f"Best ASCII:\n"
                    f"{best_ascii}"
                    f"Ep {ep} | Reward {ep_reward:.2f} | E={phys_E} | best E={best_E} | Avg Reward={avg_reward:.3f}"
                )

            # CSV logging
            row = {"episode": ep, "Reward": ep_reward, "Energy": phys_E, "Actions": ''.join(map(str, actions))}
            csv_writer.writerow(row)
            csv_file.flush()
            episode_log.append(row)

            # update tqdm postfix
            pbar.set_postfix(Reward=f"{ep_reward:.2f}", Energy=phys_E, BestE=best_E, LossV=last_loss_v, LossP=last_loss_p)

    # finalize
    torch.save(agent.nn_v.state_dict(), out_dir / "alphazero_v_weights.pt")
    torch.save(agent.nn_p.state_dict(), out_dir / "alphazero_p_weights.pt")
    pickle.dump(episode_log, (out_dir / "replay.pkl").open("wb"))
    csv_file.close()
    _plot_metrics(csv_path=csv_path, out_dir=out_dir, ma_window=args.ma_window)
    writer.close()
    logger.info(f"Run artifacts saved in {out_dir}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="HPHPPHHPHPPH")
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr_v", type=float, default=2e-4)
    parser.add_argument("--lr_p", type=float, default=2e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--update-interval", type=int, default=1000)

    parser.add_argument("--network-type", choices=["mlp","cnn", "attn"], default="mlp")
    parser.add_argument("--board-size", type=int, default=None, help="grid side length (required for CNN)")
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[256,256], help="MLP hidden sizes")
    parser.add_argument("--cnn-hidden", type=int, nargs=2, default=[128,128], help="CNN head sizes")


    parser.add_argument("--UCB-const", type=float, default=1.0)
    parser.add_argument("--MCTS-simulation-count", type=int, default=200)

    parser.add_argument(
        "--ma-window",
        type=int,
        default=500,
        help="Moving-average window size for metric plotting",
    )
    parser.add_argument("--outdir", type=str, default="runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    main(args)