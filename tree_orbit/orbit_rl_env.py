#!/usr/bin/env python3
"""
orbit_rl_env.py - compact-observation RL environment for a learned orbit controller.

Same simulator, same trees, same reward as the old OrchardEnv (it is wrapped, not
modified), but the policy does NOT see pixels. It sees what the perception block
produces, which is what a Tello can produce too:

    obs (16 floats, all roughly in [-1, 1]):
      0  target valid (0/1)          1  target bearing / 45 deg      2  target range / 10 m
      3  target blob fills frame     4  2nd tree valid               5  2nd tree bearing / 45
      6  2nd tree range / 10         7  measured body vx / 1.2       8  measured body vy / 0.8
      9  measured yaw rate / 25      10 sectors covered on target /6 11 near a finished tree (0/1)
      12 last action vx              13 last action vy              14 last action yaw
      15 wind hint: always 0 (the policy must infer wind from 7-9 vs 12-14)

    action: [vx, vy, yaw_rate] in [-1, 1], scaled exactly like the old env
            (1.2 m/s, 0.8 m/s, 25 deg/s), so the RL reward applies unchanged.

Domain randomisation at reset: wind vector (0..wind_max m/s, random heading) plus a
slow random walk during the episode, detector bearing/range noise, the layout window
and tree scales the old env already randomises.

Target bookkeeping (which tree, which sectors done) uses the same odometry-frame
tracking as the feedback controller, fed with measured velocity + yaw: nothing the
Tello cannot provide.
"""
import math
import os
import random
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bc_learning", "scripts"))

from tree_orbit import OrbitConfig, GreenBlobDetector, DepthBlobDetector  # noqa: E402
from tree_orbit.controller import Track  # noqa: E402
from tree_orbit.drones import AirSimDrone  # noqa: E402
from tree_orbit.odometry import DeadReckoning  # noqa: E402

N_SECTORS = 6


class TreeTracker:
    """Target selection + sector coverage in the odometry frame (no control)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.odom = DeadReckoning()
        self.track = None
        self.sectors = set()
        self.done = []          # finished tree positions (odom frame)
        self.t_last_seen = 0.0
        self.t = 0.0

    def reset(self):
        self.__init__(self.cfg)

    def _range_ok(self, d):
        return not d.partial

    def update(self, dets, yaw_deg, vel_body, dt):
        self.t += dt
        self.odom.set_yaw(yaw_deg)
        self.odom.step(vel_body[0], vel_body[1], dt)
        cfg = self.cfg
        if self.track is None:
            cands = [d for d in dets if not self.odom.is_done(d.bearing_deg, d.dist_m, cfg.done_tree_radius_m)
                     and d.dist_m <= cfg.max_range_m]
            if cands:
                cands.sort(key=lambda d: (d.partial, d.dist_m))
                self.track = Track(self.odom, cands[0])
                self.sectors = set()
                self.t_last_seen = self.t
            return None
        pb, pd = self.track.predict()
        best, best_cost = None, 1.0
        for d in dets:
            eb = abs(d.bearing_deg - pb) / cfg.bearing_gate_deg
            if eb > 1.0:
                continue
            ed = abs(d.dist_m - pd) / max(1.0, 0.6 * pd) if self._range_ok(d) else 0.5
            cost = 0.6 * eb + 0.4 * min(ed, 1.5)
            if cost < best_cost:
                best, best_cost = d, cost
        if best is not None:
            self.track.update(best, range_ok=self._range_ok(best))
            self.t_last_seen = self.t
        # sector coverage: angle of the drone as seen from the tree, when inside the band
        dx, dy = self.odom.x - self.track.x, self.odom.y - self.track.y
        rng = math.hypot(dx, dy)
        if 2.8 <= rng <= 5.8 and abs(pb) < 50.0:
            self.sectors.add(int(((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0) // (360.0 / N_SECTORS)))
        if len(self.sectors) >= N_SECTORS:
            self.done.append((self.track.x, self.track.y))
            self.track = None
            self.sectors = set()
        elif self.t - self.t_last_seen > cfg.lost_abort_s * 2:
            self.track = None
            self.sectors = set()
        return best

    def near_done(self, radius):
        return any(math.hypot(self.odom.x - x, self.odom.y - y) < radius for x, y in self.done)


class OrbitRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, wind_max=3.0, gust=0.4, det_noise=0.05, detector="depth", cfg=None, verbose=False,
                 action_change_penalty=0.05):
        import orchard_env as oe
        self.oe = oe
        self.cfg = cfg or OrbitConfig.sim()
        # The old env thins real layouts to 5 m between trunks; with canopies up to 1.84 m in
        # radius a 3.2 m orbit ring then grazes the neighbours. 6.5 m leaves 1.5 m clearance.
        # Set at runtime so the old file stays untouched.
        oe.LAYOUT_MIN_TREE_DIST = max(oe.LAYOUT_MIN_TREE_DIST, 6.5)
        self.inner = oe.OrchardEnv()
        self._shim_depth_requests(self.inner.client)
        self.drone = AirSimDrone(self.cfg, client=self.inner.client)
        self.det = DepthBlobDetector(self.cfg) if detector == "depth" else GreenBlobDetector(self.cfg)
        self.use_depth = detector == "depth"
        self.tracker = TreeTracker(self.cfg)
        self.wind_max, self.gust, self.det_noise = wind_max, gust, det_noise
        self.action_change_penalty = action_change_penalty
        self.verbose = verbose
        self.wind = np.zeros(2)
        self.last_action = np.zeros(3, np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._last_dets = []

    def _shim_depth_requests(self, client):
        """The old env asks for a 320x256 float depth image every step; decoding it costs
        ~640 ms. It only ever resizes it to 64x80, so serve it from the 80x64 depth camera."""
        orig = client.simGetImages
        depth_t = self.oe.airsim.ImageType.DepthPlanar

        def shim(requests, vehicle_name="", external=False):
            for r in requests:
                if r.image_type == depth_t and r.camera_name == "front_center":
                    r.camera_name = AirSimDrone.DEPTH_CAMERA
            return orig(requests, vehicle_name, external)
        client.simGetImages = shim

    # --- wind -----------------------------------------------------------------------
    def _set_wind(self, w):
        self.wind = w
        self.inner.client.simSetWind(self.oe.airsim.Vector3r(float(w[0]), float(w[1]), 0.0))

    def _random_wind(self):
        mag = random.uniform(0.0, self.wind_max)
        ang = random.uniform(0, 2 * math.pi)
        return np.array([mag * math.cos(ang), mag * math.sin(ang)])

    # --- observation ------------------------------------------------------------------
    def _observe(self):
        bgr, depth = self.drone.frame_and_depth()
        dets = self.det.detect(bgr, depth if self.use_depth else None,
                               cam_height_m=self.drone.cam_height_m(), pitch_deg=self.drone.pitch_deg())
        for d in dets:                       # detector noise (the Tello's will be worse than the sim's)
            d.bearing_deg += random.gauss(0.0, 1.5)
            d.dist_m *= 1.0 + random.gauss(0.0, self.det_noise)
        self._last_dets = dets
        full = self.inner.client.getMultirotorState()
        st = full.kinematics_estimated
        t_sim = full.timestamp * 1e-9
        self.last_dt = max(0.02, min(1.0, t_sim - self._t_sim)) if getattr(self, '_t_sim', None) else self.oe.CONTROL_DT
        self._t_sim = t_sim
        yaw = math.degrees(self.oe.airsim.to_eularian_angles(st.orientation)[2])
        vb = self.drone.vel_body()
        yaw_rate = math.degrees(st.angular_velocity.z_val)
        tgt = self.tracker.update(dets, yaw, vb, self.last_dt)
        obs = np.zeros(16, np.float32)
        if self.tracker.track is not None:
            b, r = self.tracker.track.predict()
            if tgt is not None:
                b = tgt.bearing_deg
            obs[0] = 1.0
            obs[1] = np.clip(b / 45.0, -1, 1)
            obs[2] = np.clip(r / 10.0, 0, 1)
            obs[3] = 1.0 if (tgt is not None and tgt.partial) else 0.0
            obs[10] = len(self.tracker.sectors) / N_SECTORS
        others = [d for d in dets if d is not tgt]
        if others:
            o = min(others, key=lambda d: d.dist_m)
            obs[4] = 1.0
            obs[5] = np.clip(o.bearing_deg / 45.0, -1, 1)
            obs[6] = np.clip(o.dist_m / 10.0, 0, 1)
        obs[7] = np.clip(vb[0] / 1.2, -1, 1)
        obs[8] = np.clip(vb[1] / 0.8, -1, 1)
        obs[9] = np.clip(yaw_rate / 25.0, -1, 1)
        obs[11] = 1.0 if self.tracker.near_done(self.cfg.done_tree_radius_m + 1.0) else 0.0
        obs[12:15] = self.last_action
        return obs

    # --- gym API ------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._set_wind(np.zeros(2))
        self.inner.reset()
        self.drone.home_z = self.inner.spawn_z
        self.drone.measure_ground()
        self.tracker.reset()
        self.last_action = np.zeros(3, np.float32)
        self._set_wind(self._random_wind())
        self._t_sim = None
        if self.verbose:
            print(f"[orbit-rl] wind {np.linalg.norm(self.wind):.1f} m/s")
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(3), -1, 1)
        _, r, term, trunc, info = self.inner.step(a)
        r -= self.action_change_penalty * float(np.abs(a - self.last_action).sum())
        self.last_action = a
        if self.gust > 0 and random.random() < 0.1:
            w = self.wind + np.random.normal(0, self.gust, 2)
            n = np.linalg.norm(w)
            if n > self.wind_max:
                w *= self.wind_max / n
            self._set_wind(w)
        obs = self._observe()
        info = dict(info or {})
        info["wind"] = float(np.linalg.norm(self.wind))
        info["dt_sim"] = self.last_dt
        info["trees_done_tracker"] = len(self.tracker.done)
        return obs, float(r), bool(term), bool(trunc), info

    def close(self):
        self._set_wind(np.zeros(2))
        self.inner.close()

    # helpers for eval scripts
    def score(self):
        env = self.inner
        collected = sum(1 for v in env.viewpoints if v["collected"])
        n = len(env.tree_positions_xy)
        return dict(sectors=collected, sectors_total=len(env.viewpoints),
                    trees_done=sum(1 for i in range(n) if env._is_tree_done(i)), trees=n,
                    end=env._last_done_reason)
