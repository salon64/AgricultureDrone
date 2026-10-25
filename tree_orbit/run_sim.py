#!/usr/bin/env python3
"""
run_sim.py - fly the tree-orbit controller in the AirSim orchard map.

Two modes:

  plain   (default)  trees must already be in the scene (bc_learning/scripts/place_layout.py);
                     the controller drives the drone directly. Good for watching.

  --scored           builds the old OrchardEnv (unchanged), lets it place a real layout
                     and take off, then drives it through env.step() so the SAME reward,
                     viewpoint bookkeeping and penalties score this controller exactly as
                     they score the RL policy. Prints reward, sectors collected, trees done.
                     Launch Blocks with launch_orchard.bat train for this (ClockSpeed 5).

Options:
  --detector green|depth   RGB green-blob detector (what the Tello runs) or depth blobs
  --episodes N             scored episodes (default 1)
  --max-trees N            stop after N trees (0 = all it can find / until timeout)
  --timeout S              mission time limit in simulated seconds
  --show                   OpenCV window with detections and state
  --log PATH               CSV trajectory log
"""
import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bc_learning", "scripts"))

from tree_orbit import OrbitConfig, GreenBlobDetector, DepthBlobDetector, Mission, State  # noqa: E402
from tree_orbit.detector import draw  # noqa: E402
from tree_orbit.drones import AirSimDrone  # noqa: E402


def make_detector(kind, cfg):
    return DepthBlobDetector(cfg) if kind == "depth" else GreenBlobDetector(cfg)


def run_plain(args, cfg):
    drone = AirSimDrone(cfg)
    det = make_detector(args.detector, cfg)
    mission = Mission(cfg)
    print("[sim] taking off")
    drone.takeoff()
    dt = 0.1
    t_last = None
    mission_step = 0
    logf = open(args.log, "w", newline="") if args.log else None
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["t", "state", "x", "y", "alt", "yaw", "n_dets", "tgt_dist", "tgt_bearing", "vx", "vy", "yaw_rate"])
    try:
        while mission.state != State.DONE:
            bgr, depth = drone.frame_and_depth()
            if bgr is None:
                time.sleep(0.05)
                continue
            dets = det.detect(bgr, depth if args.detector == "depth" else None,
                              cam_height_m=drone.cam_height_m(), pitch_deg=drone.pitch_deg())
            now = drone.client.getMultirotorState().timestamp * 1e-9   # simulated time: correct at any ClockSpeed
            step_dt = max(0.02, min(1.0, now - t_last)) if t_last else 0.1
            t_last = now
            cmd = mission.update(dets, drone.yaw_deg(), step_dt, vel_body=drone.vel_body())
            if drone.send(cmd, 0.6):
                mission.on_collision()
            if args.dump_dir and mission_step % 10 == 0:
                os.makedirs(args.dump_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.dump_dir, f"{mission_step:05d}.png"),
                            draw(bgr.copy(), dets, mission.target, [mission.status()]))
            mission_step += 1
            if logw:
                x, y, z, yaw = drone.pose()
                tg = mission.target_estimate() if mission.state in (State.APPROACH, State.ORBIT) else None
                logw.writerow([round(mission.t, 2), mission.state.value, round(x, 2), round(y, 2), round(drone.alt_m(), 2),
                               round(yaw, 1), len(dets), round(tg[1], 2) if tg else "", round(tg[0], 1) if tg else "",
                               round(cmd.vx, 2), round(cmd.vy, 2), round(cmd.yaw_rate, 1)])
                logf.flush()
            if args.show:
                show = cv2.resize(draw(bgr, dets, mission.target, [mission.status()]),
                                  (bgr.shape[1] * 3, bgr.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("tree_orbit (Esc quits)", show)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            time.sleep(0.02)
    finally:
        print("[sim] landing")
        drone.land()
        cv2.destroyAllWindows()
        if logf:
            logf.close()
    print(f"[sim] trees done: {mission.trees_done}  transitions: {mission.log}")


def run_scored(args, cfg):
    import orchard_env as oe
    oe.LAYOUT_MIN_TREE_DIST = max(oe.LAYOUT_MIN_TREE_DIST, 6.5)   # see orbit_rl_env.py
    env = oe.OrchardEnv()
    drone = AirSimDrone(cfg, client=env.client)
    det = make_detector(args.detector, cfg)
    results = []
    logf = open(args.log, "w", newline="") if args.log else None
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["episode", "t", "state", "x", "y", "yaw", "vx", "vy", "yaw_rate", "reward", "n_dets"])
    for ep in range(args.episodes):
        obs, _ = env.reset()
        drone.home_z = env.spawn_z
        drone.measure_ground()
        mission = Mission(cfg)
        total, steps, t = 0.0, 0, 0.0
        done = False
        t_sim_prev = None
        while not done:
            bgr, depth = drone.frame_and_depth()
            dets = det.detect(bgr, depth if args.detector == "depth" else None,
                              cam_height_m=drone.cam_height_m(), pitch_deg=drone.pitch_deg())
            t_sim = env.client.getMultirotorState().timestamp * 1e-9
            dt = max(0.02, min(1.0, t_sim - t_sim_prev)) if t_sim_prev else oe.CONTROL_DT
            t_sim_prev = t_sim
            cmd = mission.update(dets, drone.yaw_deg(), dt, vel_body=drone.vel_body())
            action = np.array([cmd.vx / oe.MAX_FORWARD_SPEED, cmd.vy / oe.MAX_SIDE_SPEED,
                               cmd.yaw_rate / oe.MAX_YAW_RATE_DEG], np.float32)
            obs, r, term, trunc, info = env.step(action)
            total += r
            steps += 1
            t += oe.CONTROL_DT
            done = term or trunc or mission.state == State.DONE
            if logw:
                x, y, z, yaw = drone.pose()
                logw.writerow([ep, round(t, 2), mission.state.value, round(x, 2), round(y, 2), round(yaw, 1),
                               round(cmd.vx, 2), round(cmd.vy, 2), round(cmd.yaw_rate, 1), round(r, 3), len(dets)])
            if args.show:
                show = cv2.resize(draw(bgr, dets, mission.target, [mission.status(), f"R {total:.1f}"]),
                                  (bgr.shape[1] * 3, bgr.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("tree_orbit scored (Esc stops)", show)
                if cv2.waitKey(1) & 0xFF == 27:
                    done = True
        collected = sum(1 for v in env.viewpoints if v["collected"])
        n_trees = len(env.tree_positions_xy)
        env_done = sum(1 for i in range(n_trees) if env._is_tree_done(i))
        res = dict(episode=ep, layout=env._layout_name, trees=n_trees, reward=round(total, 2), steps=steps,
                   sim_s=round(t, 1), sectors=f"{collected}/{len(env.viewpoints)}",
                   trees_done_env=env_done, trees_done_ctrl=mission.trees_done,
                   end=env._last_done_reason or mission.state.value)
        results.append(res)
        print(f"[scored] {res}")
        print(f"[scored] transitions: {mission.log}")
    if logf:
        logf.close()
    env.close()
    cv2.destroyAllWindows()
    if len(results) > 1:
        print(f"[scored] mean reward {np.mean([r['reward'] for r in results]):.2f}, "
              f"mean trees done (env) {np.mean([r['trees_done_env'] for r in results]):.2f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", action="store_true")
    ap.add_argument("--detector", choices=["green", "depth"], default="depth",
                    help="depth = sim depth camera (stand-in for a monocular depth model on the Tello); "
                         "green = RGB-only canopy detector with ground-plane range")
    ap.add_argument("--dump-dir", default=None, help="save an annotated frame every 10 steps here")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-trees", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=400.0)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    cfg = OrbitConfig.sim(max_trees=args.max_trees, mission_timeout_s=args.timeout)
    if args.radius:
        cfg.orbit_radius_m = args.radius
    if args.scored:
        run_scored(args, cfg)
    else:
        run_plain(args, cfg)


if __name__ == "__main__":
    main()
