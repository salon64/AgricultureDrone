#!/usr/bin/env python3
"""
run_tello.py - fly the tree-orbit controller on a DJI Tello.

Safety first: this is a 80 g drone with no obstacle sensing. Fly in an open spot,
one tree at a time, hand on the keyboard.

    python run_tello.py --dry        # camera + detector only, no take-off (check the mask!)
    python run_tello.py              # full mission

Keys in the video window:  Esc or L = land now,  Space = hover (pause/resume),
                           H = toggle HSV mask view,  Q = quit after landing.

Before a real flight:
  1. --dry and look at the detections while you hold the drone facing a tree.
     If the canopy is not boxed, tune hsv_lo/hsv_hi and crown_diameter_m in config.py
     (OrbitConfig.tello) or plug in a better detector.
  2. Check rc_per_mps: with the default 100, a 0.35 m/s orbit speed sends rc 35.
     Calibrate once by flying forward for 5 s at rc 35 and pacing the distance.
  3. Orbit radius 3 m means the drone must have 3 m of free space all the way round.
"""
import argparse
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tree_orbit import OrbitConfig, GreenBlobDetector, Mission, State  # noqa: E402
from tree_orbit.detector import draw  # noqa: E402
from tree_orbit.drones import TelloDrone  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="no take-off, just show detections")
    ap.add_argument("--max-trees", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--record", default=None, help="save annotated video to this .mp4")
    args = ap.parse_args()

    cfg = OrbitConfig.tello(max_trees=args.max_trees, mission_timeout_s=args.timeout)
    if args.radius:
        cfg.orbit_radius_m = args.radius
    drone = TelloDrone(cfg)
    det = GreenBlobDetector(cfg)
    mission = Mission(cfg)
    writer = None
    show_mask = False
    paused = False
    flying = False
    win = "tello tree_orbit  (Esc/L land, Space pause, H mask)"
    t_last = time.time()
    try:
        if not args.dry:
            print("[tello] taking off in 3 s, Esc in the window lands")
            time.sleep(3.0)
            drone.takeoff()
            flying = True
        while True:
            bgr = drone.frame()
            if bgr is None:
                time.sleep(0.02)
                continue
            dets = det.detect(bgr, cam_height_m=drone.cam_height_m(), pitch_deg=drone.pitch_deg())
            now = time.time()
            dt = max(0.02, min(0.3, now - t_last))
            t_last = now
            if flying and not paused:
                cmd = mission.update(dets, drone.yaw_deg(), dt, vel_body=drone.vel_body())
                drone.send(cmd, dt)
            elif flying:
                drone.hover()
            status = mission.status() if flying else f"DRY  dets {len(dets)}  battery {drone.tello.get_battery()}%"
            view = cv2.cvtColor(det.mask(bgr), cv2.COLOR_GRAY2BGR) if show_mask else bgr
            draw(view, dets, mission.target, [status, f"alt {drone.alt_m():.2f} m" + ("  PAUSED" if paused else "")])
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 15, (view.shape[1], view.shape[0]))
                writer.write(view)
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("l")):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("h"):
                show_mask = not show_mask
            if k == ord("q") and not flying:
                break
            if flying and mission.state == State.DONE:
                print("[tello] mission done")
                break
            if flying and time.time() - drone._last_frame_t > 2.0:
                print("[tello] video lost, landing")
                break
    finally:
        if flying:
            print("[tello] landing")
            drone.land()
        if writer is not None:
            writer.release()
        drone.close()
        cv2.destroyAllWindows()
        print(f"[tello] trees done {mission.trees_done}; transitions {mission.log}")


if __name__ == "__main__":
    main()
