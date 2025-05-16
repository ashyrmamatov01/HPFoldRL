"""scripts/train_tabular_q.py
Train a Tabular Q(λ) agent on HP2DEnv with flexible exploration options.

Usage example
-------------
python scripts/train_tabular_q.py \
    --episodes 80000 \
    --exploration ucb --ucb-c 1.4 \
    --alpha 0.7 --alpha-decay \
    --lam 0.8 \
    --max-steps 200 \
    --eval
"""
from __future__ import annotations

import argparse
import sys, signal
import logging
import json
import csv
import pickle
import pathlib
import datetime 
from typing import List, Tuple

import numpy as np
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for matplotlib

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter
# from tensorboard import SummaryWriter

from hp_problem.envs.hp2d_env import HP2DEnv
from hp_problem.agents.tabular_q import TabularQAgent
from hp_problem.utils import extract_params
from hp_problem.utils.visualize import _plot_metrics

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_and_exit(agent, episode_log: list[dict], out_dir: pathlib.Path) -> None:
    """Save intermediate artifacts on interrupt and exit."""
    try:
        with open(out_dir / "q_table_interrupt.pkl", "wb") as f:
            pickle.dump({k: v.tolist() for k, v in agent.Q.items()}, f)
        with open(out_dir / "episode_log_interrupt.pkl", "wb") as f:
            pickle.dump(episode_log, f)
        print("Saved interrupt artifacts.")
    except Exception as e:
        print(f"Error saving interrupt artifacts: {e}")
    sys.exit(0)

def run_episode(
    env: HP2DEnv,
    agent: TabularQAgent,
    rng: np.random.Generator,
    max_steps: int,
    episode: int = 0,
) -> float:
    """Play one episode, capped at *max_steps*, return final reward (energy)."""
    obs, info = env.reset(seed=int(rng.integers(1e9)))
    ep_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        action = agent.select_action(obs, info["valid_actions"])
        next_obs, reward, done, _, info2 = env.step(action)
        agent.update(obs, action, reward, next_obs, done, info2["valid_actions"])

        obs, info = next_obs, info2
        ep_reward = reward  # reward only non‑zero at episode end
        step += 1

    # print(f"\t\tEpisode finished after {step} steps with reward {ep_reward:.2f}")
    return ep_reward  # ≙ negative H–H contacts (energy)


def evaluate_greedy(
    env: HP2DEnv,
    agent: TabularQAgent,
    n_eval: int = 20,
    max_steps: int = 200,
    seed: int | None = None,
) -> Tuple[float, List[float]]:
    """Average energy of greedy policy."""
    rng = np.random.default_rng(seed)
    energies: List[float] = []
    rewards: List[float] = []
    for _ in range(n_eval):
        obs, info = env.reset(seed=int(rng.integers(1e9)))
        done, step = False, 0
        while not done and step < max_steps:
            a = agent.greedy(obs, info["valid_actions"])
            obs, r, done, _, info = env.step(a)
            step += 1
        energies.append(env._energy())
        rewards.append(r)
    return rewards, energies


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #


def main(args):
    # --- TensorBoard Setup ---
    # log_dir_tensorboard = args.out_dir / "tensorboard" # out_dir is already defined later, move its definition up
    # Define out_dir earlier
    out_dir = pathlib.Path(args.outdir).expanduser()
    # create a new directory for this run with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_dir / f"{args.sequence}" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir_tensorboard = out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir_tensorboard))
    
    
    env = HP2DEnv(sequence=args.sequence, render_mode="ascii")

    agent = TabularQAgent(
        n_actions=env.action_space.n,
        gamma=args.gamma,
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        lam=args.lam,
        exploration=args.exploration,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        # eps_decay_steps=args.episodes//2,
        total_episodes_for_decay=args.episodes,
        avg_steps_per_episode=env.n_res - 1, # Pass avg episode length (n_res - 1)
        ucb_c=args.ucb_c,
        optimistic_init=args.opt_q,
        seed=args.seed,
    )
    agent_params = extract_params(agent)
    env_params = extract_params(env)
    # save configs to JSON
    configs = {
        "agent": agent_params,
        "env": env_params,
        "args": vars(args),
    }
    args_json_path = out_dir / "args.json"
    with open(args_json_path, "w") as f:
        json.dump(configs, f, indent=4)
        print(f"Saved configs to {args_json_path}")

    # prepare CSV logging
    csv_path = out_dir / "training_log.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["episode", "Reward", "Energy", "epsilon", "BestEnergy", "Outcome"])
    csv_writer.writeheader()

    rng = np.random.default_rng(args.seed)
    rewards: List[dict] = []
    episode_log = []
    avg_E_interval = float("nan")

    # safely exit on interrupt
    signal.signal(signal.SIGINT, lambda *_: save_and_exit(agent, episode_log, out_dir))

    best_energy_so_far = float("inf")
    best_coords_so_far = None
    best_sequence_str = "".join(args.sequence)

    # -------------------- serial training ------------------ #
    with tqdm(total=args.episodes, file=sys.stdout, desc="Training", unit='ep') as pbar:
        # for episode in trange(args.episodes, desc="Training"):
        for episode in range(args.episodes):
            episode_reward = run_episode(env, agent, rng, args.max_steps, episode)
            rewards.append(episode_reward)

            # coords = env.get_coords()
            physical_energy_of_this_episode = env._energy()
            _epsilon = agent._epsilon()
            info_str_from_run = env.info_str_from_last_step 

            writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
            writer.add_scalar("Agent/epsilon", _epsilon, episode)
            writer.add_scalar("Agent/GlobalStep", agent.global_step, episode)

            if env.step_idx == env.n_res and info_str_from_run == 'completed':
                writer.add_scalar("Train/Energy", physical_energy_of_this_episode, episode)
                if physical_energy_of_this_episode < best_energy_so_far:
                    best_energy_so_far = physical_energy_of_this_episode
                    best_coords_so_far = env.get_coords()
                    best_coords_ascii = env.render(mode="ascii", legend=True)

            # episode_data_for_cvs.append({
            #     "episode": episode+1,
            #     "Reward": episode_reward,
            #     "Energy": physical_energy_of_this_episode,
            #     "epsilon": _epsilon,
            #     "BestEnergy": best_energy_so_far if best_energy_so_far != float("inf") else float('nan'),
            #     "Outcome": info_str_from_run,
            # })
            row ={
                 "episode": episode+1,
                 "Reward": episode_reward,
                 "Energy": physical_energy_of_this_episode,
                 "epsilon": _epsilon,
                 "BestEnergy": best_energy_so_far if best_energy_so_far != float("inf") else float('nan'),
                 "Outcome": info_str_from_run,
             }
            csv_writer.writerow(row)
            csv_file.flush()
            episode_log.append(row)

            if (episode + 1) % args.log_interval == 0:
                avg_env_reward_interval = float(np.mean(rewards[-args.log_interval :])) # Average of rewards agent sees
                writer.add_scalar("Train/AvgEnvReward_LogInterval", avg_env_reward_interval, episode)

                # Render current state if desired (optional)
                current_render_str = env.render(mode="ascii", legend=True)
                tqdm.write(f"Current ep structure (Outcome: {info_str_from_run}):\n{current_render_str}")
                tqdm.write(f"Best physical energy structure:\n{best_coords_ascii}")
                
                # Prepare log message parts
                log_message_parts = [
                    f"Ep {episode + 1:>6}",
                    f"AvgEnvRew={avg_env_reward_interval:6.2f}"
                ]
                if best_energy_so_far != float("inf"):
                    log_message_parts.append(f"BestPhysE={best_energy_so_far:.2f}") # Display best *physical* energy
                else:
                    log_message_parts.append("BestPhysE=N/A")
                log_message_parts.append(f"eps={_epsilon:.3f}")
                log_message_parts.append(f"LastOutcome={info_str_from_run}")
                tqdm.write(" | ".join(log_message_parts))

            pbar.update(1)
            pbar.set_postfix({
                "BestPhysE": f"{best_energy_so_far if best_energy_so_far != float('inf') else 'N/A':.2f}", 
                "LastEnvRew": f"{episode_reward:.2f}", # Show the reward the agent got
                "LastPhysE": f"{physical_energy_of_this_episode:.2f}" if info_str_from_run == 'completed' else info_str_from_run[:5], # Show physical energy if completed
                # "Outcome": info_str_from_run[:10] # Already covered by LastPhysE essentially
            })

    # -------------------- parallel training ------------------ #
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [
    #         executor.submit(run_episode, env, agent, rng, args.max_steps)
    #         for _ in range(args.episodes)
    #     ]
    #     for i, f in enumerate(as_completed(futures)):
    #         episode_energies = f.result();
    #         energies.append(episode_energies);
    #         # if (i + 1) % args.log_interval == 0:
    #         if True:
    #             avg_E = float(np.mean(energies[-args.log_interval :]))
    #             print(
    #                 f"Ep {i+1:>6} | avg energy {avg_E:6.2f} | exploration={agent.exploration} | eps={agent._epsilon():.3f}"
    #             )
    

    # -------------------- save artifacts -------------------- #
    writer.close()
    csv_writer.close()
    _plot_metrics(
        csv_path=csv_path,
        out_dir=out_dir,
        ma_window=args.ma_window,
    )
    (out_dir / "train_rewards.json").write_text(json.dumps(rewards))
    with open(out_dir / "q_table.pkl", "wb") as f:
        pickle.dump({k: v.tolist() for k, v in agent.Q.items()}, f)

    # -------------------- optional evaluation --------------- #
    if args.eval:
        all_R, all_E = evaluate_greedy(env, agent, n_eval=args.eval_episodes, max_steps=args.max_steps, seed=args.seed)
        mean_E = np.mean(all_E)
        mean_R = np.mean(all_R)
        print(f"Greedy policy mean reward over {len(all_R)} runs: {mean_R:.2f}")
        print(f"Greedy policy mean energy over {len(all_E)} runs: {mean_E:.2f}")
        (out_dir / "eval_energies.json").write_text(json.dumps(all_E))
        (out_dir / "eval_rewards.json").write_text(json.dumps(all_R))

    print(f"Artifacts saved in {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=str, default="HPHPPHHPHPPH", help="sequence to fold")
    p.add_argument("--episodes", type=int, default=100_000)
    p.add_argument("--max-steps", type=int, default=300, help="episode length cap")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--alpha-decay", action="store_true")
    p.add_argument("--lam", type=float, default=0.0, help="eligibility trace λ")

    # exploration flags
    p.add_argument("--exploration", choices=["eps", "ucb"], default="eps")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    # p.add_argument("--eps-decay", type=int, default=10_000)
    p.add_argument("--ucb-c", type=float, default=1.2)

    p.add_argument("--opt-q", type=float, default=0.0, help="optimistic Q init")
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--outdir", type=str, default="runs/tabular_q", help="output directory")
    p.add_argument("--eval", action="store_true", help="evaluate greedy at end")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    
    # plotting options
    p.add_argument("--ma-window", type=int, default=500, help="window size for moving average")

    main(p.parse_args())
