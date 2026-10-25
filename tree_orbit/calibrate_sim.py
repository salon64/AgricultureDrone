#!/usr/bin/env python3
"""
calibrate_sim.py - solve the detector's canopy_bottom_m from ground truth in the sim.

The RGB range estimate is  range = (cam_height - canopy_bottom) / tan(angle below horizon)
where the angle comes from the lowest canopy pixel of the blob. canopy_bottom_m is the
height of that lowest *visible-in-the-mask* pixel, which depends on lighting, not on
the mesh, so it is measured rather than assumed. On a real tree, measure it with a
tape (lowest leaves) and set OrbitConfig.tello(canopy_bottom_m=...).

Usage (Blocks running, a layout placed):
    python calibrate_sim.py            -> prints the fitted value and per-sample errors
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tree_orbit import OrbitConfig, GreenBlobDetector  # noqa: E402
from tree_orbit.detector import focal_px  # noqa: E402
from tree_orbit.drones import AirSimDrone  # noqa: E402


def main():
    cfg = OrbitConfig.sim()
    d = AirSimDrone(cfg)
    g = GreenBlobDetector(cfg)
    trees = [(n, d.client.simGetObjectPose(n).position) for n in d.client.simListSceneObjects("Tree.*")]
    trees = [(n, p.x_val, p.y_val) for n, p in trees if p.x_val > -100]
    if not trees:
        sys.exit("no trees in the scene; run bc_learning/scripts/place_layout.py first")
    # pick the tree with the most free space to its west (we approach from -x)
    name, tx, ty = max(trees, key=lambda t: min([99] + [math.hypot(t[1] - o[1], t[2] - o[2]) for o in trees if o is not t]))
    print(f"[cal] using {name} at ({tx:.1f}, {ty:.1f})")
    d.takeoff()
    f = focal_px(cfg.frame_w, cfg.hfov_deg)
    samples = []
    for rng in (9.0, 7.0, 5.5, 4.2, 3.4):
        d.client.moveToPositionAsync(tx - rng, ty, d.home_z - cfg.cruise_alt_m, 1.0, timeout_sec=20).join()
        time.sleep(0.8)
        d.client.rotateToYawAsync(0, 5).join()
        time.sleep(1.0)
        bgr, depth = d.frame_and_depth()
        ch, pitch = d.cam_height_m(), d.pitch_deg()
        m = g.mask(bgr)
        dets = g.detect(bgr, cam_height_m=ch, pitch_deg=pitch)
        if not dets:
            continue
        t = min(dets, key=lambda q: abs(q.bearing_deg))
        x, y, w, h = t.bbox
        c0, c1 = int(max(x, x + 0.35 * w)), int(x + 0.65 * w) + 1
        rows = np.nonzero(m[y:y + h, c0:c1].any(axis=1))[0]
        bottom = y + rows.max() + 0.5
        ang = math.atan2(bottom - cfg.frame_h / 2.0, f) - math.radians(pitch)
        p = d.client.simGetVehiclePose().position
        true = math.hypot(tx - p.x_val, ty - p.y_val) * math.cos(math.radians(t.bearing_deg))
        canopy_bottom = ch - true * math.tan(ang)
        samples.append((true, ch, math.degrees(ang), canopy_bottom))
        print(f"[cal] true {true:5.2f} m  cam_h {ch:.2f}  bottom angle {math.degrees(ang):5.2f} deg  -> canopy_bottom {canopy_bottom:.2f} m")
    d.land()
    if len(samples) >= 2:
        cb = float(np.median([s[3] for s in samples]))
        print(f"\n[cal] canopy_bottom_m = {cb:.2f}   (set this in OrbitConfig.sim)")
        for true, ch, ang, _ in samples:
            est = (ch - cb) / math.tan(math.radians(ang))
            print(f"      true {true:5.2f} -> est {est:5.2f} ({(est / true - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    main()
