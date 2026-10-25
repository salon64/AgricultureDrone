#!/usr/bin/env python3
"""
eval_wind.py - score controllers under wind with the SAME environment and reward.

    python eval_wind.py --controller feedback --winds 0 2 4 --episodes 3
    python eval_wind.py --controller ppo --model ~/bc_data/orbit_rl/orbit_ppo_final.zip --winds 0 2 4
    python eval_wind.py --controller both ...

Both controllers get the compact observation pipeline of orbit_rl_env.py; the feedback
controller (tree_orbit.controller.Mission) additionally sees the raw detections it was
written for. Reported per wind level: mean reward, sectors collected, trees finished.
Launch Blocks with launch_orchard.bat train first.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tree_orbit import OrbitConfig, Mission  # noqa: E402
from tree_orbit.orbit_rl_env import OrbitRLEnv  # noqa: E402


def run_feedback(env, cfg):
    import orchard_env as oe
    mission = Mission(cfg)
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        dets = env._last_dets
        cmd = mission.update(dets, env.drone.yaw_deg(), getattr(env, 'last_dt', oe.CONTROL_DT),
                             vel_body=env.drone.vel_body())
        a = np.array([cmd.vx / oe.MAX_FORWARD_SPEED, cmd.vy / oe.MAX_SIDE_SPEED,
                      cmd.yaw_rate / oe.MAX_YAW_RATE_DEG], np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    return total, env.score()


def run_ppo(env, model):
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    return total, env.score()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["feedback", "ppo", "both"], default="feedback")
    ap.add_argument("--model", default=os.path.expanduser("~/bc_data/orbit_rl/orbit_ppo_final.zip"))
    ap.add_argument("--winds", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--detector", choices=["green", "depth"], default="depth")
    args = ap.parse_args()

    cfg = OrbitConfig.sim()
    env = OrbitRLEnv(wind_max=0.0, gust=0.0, det_noise=0.05, detector=args.detector, cfg=cfg)
    model = None
    if args.controller in ("ppo", "both"):
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
    rows = []
    for wind in args.winds:
        env.wind_max = wind
        env.gust = 0.3 if wind > 0 else 0.0
        for name in (["feedback", "ppo"] if args.controller == "both" else [args.controller]):
            rs, secs, trees = [], [], []
            for ep in range(args.episodes):
                # fixed-magnitude wind at this level, random heading (set after reset's own draw)
                env._random_wind = lambda w=wind: (lambda ang: np.array([w * np.cos(ang), w * np.sin(ang)]))(np.random.uniform(0, 2 * np.pi))
                total, sc = run_feedback(env, cfg) if name == "feedback" else run_ppo(env, model)
                rs.append(total); secs.append(sc["sectors"] / max(1, sc["sectors_total"])); trees.append(sc["trees_done"])
                print(f"[eval] wind {wind:.1f} {name:8s} ep {ep}: reward {total:7.2f}  sectors {sc['sectors']}/{sc['sectors_total']}  "
                      f"trees {sc['trees_done']}/{sc['trees']}  end {sc['end']}")
            rows.append((wind, name, np.mean(rs), np.mean(secs), np.mean(trees)))
    print("\nwind  controller  mean_reward  sector_frac  trees_done")
    for wind, name, r, s, t in rows:
        print(f"{wind:4.1f}  {name:10s} {r:11.2f}  {s:11.2f}  {t:10.2f}")
    env.close()


if __name__ == "__main__":
    main()
