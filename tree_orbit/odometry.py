"""
Dead-reckoning odometry + memory of finished trees.

The Tello has no position sensor, so position is integrated from the commanded body
velocity and the IMU yaw. It drifts, but it only has to be good enough to recognise
"that tree 4 m away is the one I just finished" for a minute or two. In AirSim the
same class is used (optionally fed with ground-truth pose for debugging).

Frame: x forward along the initial heading, y to the right, yaw clockwise-positive
in degrees (AirSim NED and the Tello rc convention agree on this).
"""
import math


class DeadReckoning:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw_deg = 0.0
        self.done_trees = []      # [(x, y), ...] in this frame
        self._yaw_unwrapped = None
        self._last_raw_yaw = None

    # --- yaw bookkeeping (unwrapped so orbit progress can be integrated) ------------
    def set_yaw(self, raw_yaw_deg):
        if self._last_raw_yaw is None:
            self._yaw_unwrapped = raw_yaw_deg
        else:
            d = (raw_yaw_deg - self._last_raw_yaw + 180.0) % 360.0 - 180.0
            self._yaw_unwrapped += d
        self._last_raw_yaw = raw_yaw_deg
        self.yaw_deg = raw_yaw_deg

    @property
    def yaw_unwrapped(self):
        return self._yaw_unwrapped if self._yaw_unwrapped is not None else 0.0

    # --- position --------------------------------------------------------------------
    def step(self, vx_body, vy_body, dt):
        r = math.radians(self.yaw_deg)
        self.x += (vx_body * math.cos(r) - vy_body * math.sin(r)) * dt
        self.y += (vx_body * math.sin(r) + vy_body * math.cos(r)) * dt

    def set_pose(self, x, y, yaw_deg):
        """Override with an external estimate (AirSim ground truth, a tracker, ...)."""
        self.x, self.y = x, y
        self.set_yaw(yaw_deg)

    # --- trees ------------------------------------------------------------------------
    def project(self, bearing_deg, dist_m):
        """World (x, y) of something seen at bearing/dist from the current pose."""
        a = math.radians(self.yaw_deg + bearing_deg)
        return self.x + dist_m * math.cos(a), self.y + dist_m * math.sin(a)

    def mark_done(self, bearing_deg, dist_m):
        self.done_trees.append(self.project(bearing_deg, dist_m))

    def is_done(self, bearing_deg, dist_m, radius_m):
        px, py = self.project(bearing_deg, dist_m)
        return any(math.hypot(px - tx, py - ty) < radius_m for tx, ty in self.done_trees)
