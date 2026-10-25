#!/usr/bin/env python3
"""
train_orbit_rl.py - PPO on the compact orbit observation (orbit_rl_env.py).

A 16-number observation and a 2-layer MLP: this trains in hours, not days, and the
policy carries over to the Tello because every input is something the Tello can
measure. Wind, gusts and detector noise are randomised every episode so the policy
has to cope with conditions it has not seen exactly.

    bc_learning\\airsim\\launch_orchard.bat train          (ClockSpeed 5)
    python train_orbit_rl.py --timesteps 300000 --wind-max 3.0
    python train_orbit_rl.py --resume                    (newest checkpoint)

Checkpoints: ~/bc_data/orbit_rl/orbit_ppo_<steps>_steps.zip
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SAVE_DIR = os.path.expanduser("~/bc_data/orbit_rl")
NAME = "orbit_ppo"


def latest_checkpoint():
    best, steps = None, -1
    for p in glob.glob(os.path.join(SAVE_DIR, f"{NAME}_*_steps.zip")):
        m = re.search(r"_(\d+)_steps\.zip$", p)
        if m and int(m.group(1)) > steps:
            best, steps = p, int(m.group(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--wind-max", type=float, default=3.0)
    ap.add_argument("--gust", type=float, default=0.4)
    ap.add_argument("--det-noise", type=float, default=0.05)
    ap.add_argument("--detector", choices=["green", "depth"], default="depth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=10_000)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from tree_orbit.orbit_rl_env import OrbitRLEnv

    os.makedirs(SAVE_DIR, exist_ok=True)
    env = Monitor(OrbitRLEnv(wind_max=args.wind_max, gust=args.gust, det_noise=args.det_noise,
                             detector=args.detector, verbose=True),
                  filename=os.path.join(SAVE_DIR, "monitor.csv"))
    ckpt = latest_checkpoint() if args.resume else None
    if ckpt:
        print(f"[train] resuming {ckpt}")
        model = PPO.load(ckpt, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=1,
                    n_steps=512, batch_size=128, n_epochs=10, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    tensorboard_log=os.path.join(SAVE_DIR, "tb"))
    cb = CheckpointCallback(save_freq=args.save_every, save_path=SAVE_DIR, name_prefix=NAME, verbose=1)
    print(f"[train] PPO on compact obs, wind up to {args.wind_max} m/s, {args.timesteps} steps. Ctrl+C saves.")
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb, reset_num_timesteps=not ckpt)
    except KeyboardInterrupt:
        print("[train] interrupted")
    final = os.path.join(SAVE_DIR, f"{NAME}_final")
    model.save(final)
    print(f"[train] saved {final}.zip")
    env.close()


if __name__ == "__main__":
    main()
