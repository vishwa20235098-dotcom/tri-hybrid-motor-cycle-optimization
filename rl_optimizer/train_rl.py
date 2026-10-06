"""
train_rl.py

Trains ONE PPO agent across multiple driving cycles at once (all registered Indian cycles by
default). Each parallel env instance randomly samples a different cycle on
every episode reset, so the resulting policy is not overfit to a single
cycle's rhythm -- it has to learn a power-split strategy that works for
both.

Run traffic_scenario.py first to generate the *_traffic.csv files.

Usage:
    python generate_pune_cycle.py               # (only needed once, builds PUNE_DC.csv)
    python traffic_scenario.py                  # builds CMDC_traffic.csv + PUNE_traffic.csv
    python train_rl.py --timesteps 300000 --cycles cmdc pune
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import TriHybridEnv
from cycles import load_cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--cycles", nargs="+", default=["cmdc", "pune", "delhi"],
                     help="Which cycles to train on jointly (default: cmdc pune delhi)")
    ap.add_argument("--out", type=str, default="ems_ppo_model.zip")
    args = ap.parse_args()

    cycles = load_cycles(names=args.cycles)
    durations = ", ".join(f"{c['name']}={c['t'][-1]:.0f}s" for c in cycles)
    print(f"Training jointly on cycles: {durations}")

    def env_fn():
        return TriHybridEnv(cycles)

    vec_env = make_vec_env(env_fn, n_envs=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
