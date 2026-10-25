"""
Mission state machine + orbit controller. Pure logic, no I/O: feed it detections,
the current yaw and dt; it returns a body-frame velocity command.

    SEARCH   rotate in place until an unfinished tree is seen
    APPROACH keep the tree centred, drive to the orbit ring
    ORBIT    strafe sideways around the tree while keeping it centred; the heading
             turns with the orbit, so accumulated yaw == orbit angle. 380 deg -> done
    LEAVE    back out and turn away so the finished tree drops out of view, then SEARCH
    DONE     max_trees reached or mission timeout

The target is tracked as a point in the dead-reckoning frame (odometry.py), not just
as "the blob nearest to where it was". Each frame the track predicts where the tree
should appear (bearing and range); the detection that best matches both is used to
update the track. That stops the controller hopping between neighbouring trees when
the range from blob width is noisy, which it is whenever canopies overlap in the image.
"""
import enum
import math
from dataclasses import dataclass

from .config import OrbitConfig
from .odometry import DeadReckoning


@dataclass
class Command:
    vx: float = 0.0          # m/s forward (body)
    vy: float = 0.0          # m/s right (body)
    vz: float = 0.0          # m/s down (NED); backends convert
    yaw_rate: float = 0.0    # deg/s, clockwise positive

    def clipped(self, cfg):
        s = cfg.max_speed_mps
        return Command(max(-s, min(s, self.vx)), max(-s, min(s, self.vy)),
                       max(-s, min(s, self.vz)),
                       max(-cfg.max_yaw_rate_dps, min(cfg.max_yaw_rate_dps, self.yaw_rate)))


class State(enum.Enum):
    SEARCH = "search"
    APPROACH = "approach"
    ORBIT = "orbit"
    LEAVE = "leave"
    DONE = "done"


class Track:
    """One tree, as a point in the odometry frame, smoothed over detections."""

    def __init__(self, odom, det, alpha=0.35):
        self.odom = odom
        self.alpha = alpha
        self.x, self.y = odom.project(det.bearing_deg, det.dist_m)
        self.last_det = det
        self.bearing = det.bearing_deg
        self.dist = det.dist_m

    def predict(self):
        dx, dy = self.x - self.odom.x, self.y - self.odom.y
        rng = math.hypot(dx, dy)
        brg = (math.degrees(math.atan2(dy, dx)) - self.odom.yaw_deg + 180.0) % 360.0 - 180.0
        return brg, rng

    def update(self, det, range_ok=True):
        pb, pd = self.predict()
        d = det.dist_m if range_ok else pd
        nx, ny = self.odom.project(det.bearing_deg, d)
        a = self.alpha if range_ok else self.alpha * 0.5
        self.x += a * (nx - self.x)
        self.y += a * (ny - self.y)
        self.last_det = det
        self.bearing = det.bearing_deg          # bearing is reliable: use it raw
        self.dist = self.predict()[1]           # range: use the smoothed track


class Mission:
    def __init__(self, cfg: OrbitConfig, odom: DeadReckoning = None, orbit_clockwise=True):
        self.cfg = cfg
        self.odom = odom or DeadReckoning()
        self.state = State.SEARCH
        self.t = 0.0
        self.t_state = 0.0
        self.track = None
        self.target = None            # last matched Detection (for drawing)
        self.t_last_seen = None
        self.orbit_progress = 0.0     # yaw travelled in the orbit direction around the current tree
        self._bumped_t = -1e9
        self.t_last_flip = -1e9
        self.orbit_sign = 1.0 if orbit_clockwise else -1.0   # +1: strafe right, heading turns left
        self.trees_done = 0
        self.search_dir = 1.0
        self.last_cmd = Command()
        self.log = []

    def _yaw_cmd(self, bearing):
        """Proportional yaw with a deadband so detector jitter does not become head-snaps."""
        db = self.cfg.yaw_deadband_deg
        err = 0.0 if abs(bearing) < db else bearing - math.copysign(db, bearing)
        return self.cfg.k_yaw * err

    def _slew(self, cmd, dt):
        """Limit how fast the commanded velocity and yaw rate may change."""
        dv = self.cfg.max_accel_mps2 * dt
        dw = self.cfg.max_yaw_accel_dps2 * dt
        p = self.last_cmd
        lim = lambda new, old, d: old + max(-d, min(d, new - old))  # noqa: E731
        return Command(lim(cmd.vx, p.vx, dv), lim(cmd.vy, p.vy, dv), lim(cmd.vz, p.vz, dv),
                       lim(cmd.yaw_rate, p.yaw_rate, dw))

    # ---------------------------------------------------------------------------------
    def _enter(self, s):
        self.log.append((round(self.t, 1), self.state.value, "->", s.value))
        self.state = s
        self.t_state = 0.0

    def _range_ok(self, det):
        """Width-based range is garbage when the blob is cut by the frame edge."""
        return not det.partial

    def _pick_new_target(self, dets):
        cands = [d for d in dets
                 if not self.odom.is_done(d.bearing_deg, d.dist_m, self.cfg.done_tree_radius_m)
                 and d.dist_m <= self.cfg.max_range_m]
        if not cands:
            return None
        cands.sort(key=lambda d: (d.partial, d.dist_m))
        return cands[0]

    def _associate(self, dets):
        """Detection that best matches the track's predicted bearing AND range."""
        if self.track is None:
            return None
        pb, pd = self.track.predict()
        best, best_cost = None, 1.0
        for d in dets:
            eb = abs(d.bearing_deg - pb) / self.cfg.bearing_gate_deg
            if eb > 1.0:
                continue
            ed = abs(d.dist_m - pd) / max(1.0, 0.6 * pd) if self._range_ok(d) else 0.5
            cost = 0.6 * eb + 0.4 * min(ed, 1.5)
            if cost < best_cost:
                best, best_cost = d, cost
        return best

    # ---------------------------------------------------------------------------------
    def update(self, dets, yaw_deg, dt, vel_body=None):
        """dets: list[Detection]; yaw_deg: raw heading from the drone; dt: seconds;
        vel_body: measured (vx, vy) m/s in the body frame if the drone reports one
        (AirSim state, Tello optical-flow speed). Without it the commanded velocity is
        integrated, which drifts badly whenever the drone is blocked."""
        cfg = self.cfg
        first = self.odom._last_raw_yaw is None
        prev_unwrapped = self.odom.yaw_unwrapped
        self.odom.set_yaw(yaw_deg)
        yaw_delta = 0.0 if first else self.odom.yaw_unwrapped - prev_unwrapped
        self.t += dt
        self.t_state += dt

        if self.state != State.DONE and self.t > cfg.mission_timeout_s:
            self._enter(State.DONE)

        cmd = Command()

        if self.state == State.DONE:
            pass

        elif self.state == State.SEARCH:
            cmd.yaw_rate = self.search_dir * cfg.search_yaw_rate_dps
            tgt = self._pick_new_target(dets)
            if tgt is not None:
                self.track = Track(self.odom, tgt)
                self.target, self.t_last_seen = tgt, self.t
                self._enter(State.APPROACH)

        elif self.state in (State.APPROACH, State.ORBIT):
            d = self._associate(dets)
            if d is not None:
                self.track.update(d, range_ok=self._range_ok(d))
                self.target, self.t_last_seen = d, self.t
            lost_for = self.t - self.t_last_seen
            bearing, dist = self.track.predict()
            if d is not None:
                bearing = d.bearing_deg
            # blob size is not a safe proximity cue (merged blobs span the frame); range is
            too_close = self._bumped_t > self.t - 2.0

            if lost_for > cfg.lost_abort_s:
                self.search_dir = 1.0 if bearing >= 0 else -1.0
                self.track, self.target = None, None
                self._enter(State.SEARCH)
            elif lost_for > cfg.lost_grace_s:
                cmd.yaw_rate = self._yaw_cmd(bearing)       # turn to where it should be
            else:
                dist_err = dist - cfg.orbit_radius_m
                cmd.yaw_rate = self._yaw_cmd(bearing)
                cmd.vx = cfg.k_dist * dist_err
                if dist < cfg.min_safe_dist_m or too_close:
                    cmd.vx = -cfg.approach_speed_mps
                if self.state == State.APPROACH:
                    cmd.vx = max(-cfg.approach_speed_mps, min(cfg.approach_speed_mps, cmd.vx))
                    # a different tree close and ahead: slide sideways past it instead of pushing
                    ahead = [o for o in dets if o is not d and o.dist_m < cfg.obstacle_dist_m + 0.4
                             and abs(o.bearing_deg) < 35.0]
                    if ahead:
                        o = min(ahead, key=lambda q: q.dist_m)
                        cmd.vy = -math.copysign(cfg.orbit_speed_mps, o.bearing_deg if o.bearing_deg != 0 else 1.0)
                        cmd.vx = min(cmd.vx, 0.0)
                    if abs(dist_err) < cfg.radius_tol_m and abs(bearing) < 12.0 and not too_close:
                        self.orbit_progress = 0.0
                        self._enter(State.ORBIT)
                else:
                    # another tree in the way on the side we are strafing to? go round the other way
                    blockers = [o for o in dets if o is not d and o.dist_m < cfg.obstacle_dist_m
                                and o.bearing_deg * self.orbit_sign > 8.0]
                    if blockers and self.t - self.t_last_flip > cfg.flip_cooldown_s:
                        self.orbit_sign = -self.orbit_sign
                        self.t_last_flip = self.t
                        self.log.append((round(self.t, 1), "flip",
                                         f"blocker {blockers[0].dist_m:.1f}m {blockers[0].bearing_deg:+.0f}deg"))
                    # heading turns against the strafe direction on a real orbit; oscillating in
                    # place (stuck on a canopy) nets to zero instead of counting as progress
                    self.orbit_progress += -self.orbit_sign * yaw_delta
                    # strafe around the ring; feed-forward the heading rate the orbit needs
                    cmd.vy = self.orbit_sign * cfg.orbit_speed_mps
                    ff = -self.orbit_sign * math.degrees(cfg.orbit_speed_mps / cfg.orbit_radius_m)
                    cmd.yaw_rate = ff + self._yaw_cmd(bearing)
                    cmd.vx = max(-cfg.orbit_speed_mps, min(cfg.orbit_speed_mps, cmd.vx))
                    if too_close:
                        cmd.vx, cmd.vy = -cfg.approach_speed_mps, 0.0
                    if self.orbit_progress >= cfg.orbit_total_deg:
                        self.odom.done_trees.append((self.track.x, self.track.y))
                        self.trees_done += 1
                        self.log.append((round(self.t, 1), f"tree {self.trees_done} done"))
                        if cfg.max_trees and self.trees_done >= cfg.max_trees:
                            self._enter(State.DONE)
                        else:
                            self._enter(State.LEAVE)

        elif self.state == State.LEAVE:
            fx, fy = self.odom.done_trees[-1]
            fb = (math.degrees(math.atan2(fy - self.odom.y, fx - self.odom.x)) - self.odom.yaw_deg + 180.0) % 360.0 - 180.0
            fd = math.hypot(fx - self.odom.x, fy - self.odom.y)
            if self.t_state < 2.0 and fd < cfg.orbit_radius_m + 1.0:
                cmd.vx = -cfg.approach_speed_mps          # open the gap first
            cmd.yaw_rate = -self.orbit_sign * cfg.search_yaw_rate_dps * 1.5
            if (self.t_state > 1.0 and abs(fb) > cfg.leave_clear_deg) or self.t_state >= cfg.leave_s:
                self.search_dir = -self.orbit_sign
                self.track, self.target = None, None
                self._enter(State.SEARCH)

        cmd = self._slew(cmd.clipped(cfg), dt)
        if vel_body is not None:
            self.odom.step(vel_body[0], vel_body[1], dt)
        else:
            self.odom.step(cmd.vx, cmd.vy, dt)
        self.last_cmd = cmd
        return cmd

    # ---------------------------------------------------------------------------------
    def on_collision(self):
        """Backend says we bumped into something: assume the target is much closer than
        we think and back off for a couple of seconds."""
        self._bumped_t = self.t
        if self.track is not None:
            b, _ = self.track.predict()
            self.track.x, self.track.y = self.odom.project(b, 1.5)

    def orbit_progress_deg(self):
        return self.orbit_progress if self.state == State.ORBIT else 0.0

    def target_estimate(self):
        """(bearing_deg, dist_m) of the tracked tree, or None."""
        return self.track.predict() if self.track is not None else None

    def status(self):
        s = f"{self.state.value.upper():8s} trees {self.trees_done}  t {self.t:5.0f}s"
        if self.track is not None and self.state in (State.APPROACH, State.ORBIT):
            b, r = self.track.predict()
            s += f"  tgt {r:.1f}m {b:+.0f}deg"
        if self.state == State.ORBIT:
            s += f"  orbit {self.orbit_progress_deg():.0f}/{self.cfg.orbit_total_deg:.0f}"
        return s
