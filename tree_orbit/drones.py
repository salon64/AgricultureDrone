"""
Drone backends with one tiny interface:

    frame()      -> BGR image (numpy)           depth() -> float32 metres or None
    yaw_deg()    -> heading, clockwise positive  alt_m() -> height above take-off
    send(cmd, dt)                                takeoff() / land() / close()

AirSimDrone  - talks to the running Blocks map (same client the old env uses).
TelloDrone   - DJI Tello over Wi-Fi via djitellopy. No depth, no position.
"""
import math
import time

import numpy as np

from .config import OrbitConfig
from .controller import Command


class AirSimDrone:
    DEPTH_CAMERA = "depth_lowres"   # 80x64: a full-res float depth image costs ~640 ms over RPC

    def __init__(self, cfg: OrbitConfig, client=None, camera="front_center"):
        import airsim
        self.airsim = airsim
        self.cfg = cfg
        self.camera = camera
        self.client = client or airsim.MultirotorClient()
        self.client.confirmConnection()
        info = self.client.simGetCameraInfo(camera)
        self.cfg.hfov_deg = float(info.fov)
        self.home_z = self.client.simGetVehiclePose().position.z_val

    def frame(self):
        r = self.client.simGetImages([self.airsim.ImageRequest(self.camera, self.airsim.ImageType.Scene, False, False)])[0]
        if r.width == 0:
            return None
        img = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)
        return img.copy()   # AirSim gives BGR already

    def depth(self):
        r = self.client.simGetImages([self.airsim.ImageRequest(self.DEPTH_CAMERA, self.airsim.ImageType.DepthPlanar, True, False)])[0]
        if r.width == 0:
            return None
        return np.array(r.image_data_float, np.float32).reshape(r.height, r.width)

    def _depth_ground_height(self):
        """Camera height above ground from the depth image's bottom rows. Only valid when
        nothing but ground is in the bottom of the frame, so it is used once at take-off."""
        d = getattr(self, '_last_depth', None)
        if d is None:
            return None
        h, w = d.shape
        f = (w / 2.0) / math.tan(math.radians(self.cfg.hfov_deg) / 2.0)
        pitch = math.radians(self.pitch_deg())
        rows = d[h - 4:h, int(0.3 * w):int(0.7 * w)]
        dist = float(np.percentile(rows[np.isfinite(rows)], 70))
        ang = math.atan2(h / 2.0 - 2.0, f) - pitch
        return max(0.3, dist * math.tan(ang))

    def cam_height_m(self):
        """Camera height above ground. The ground's NED z is measured once at take-off over
        clear ground (flat map); afterwards it is pose-based, like the Tello's downward
        ToF sensor, and cannot be fooled by a tree in the forward view."""
        gz = getattr(self, 'ground_z_ned', None)
        if gz is None:
            return -(self.pose()[2] - self.home_z) + 0.05
        return gz - self.pose()[2]

    def pitch_deg(self):
        p = self.client.simGetVehiclePose()
        return math.degrees(self.airsim.to_eularian_angles(p.orientation)[0])

    def frame_and_depth(self):
        rs = self.client.simGetImages([
            self.airsim.ImageRequest(self.camera, self.airsim.ImageType.Scene, False, False),
            self.airsim.ImageRequest(self.DEPTH_CAMERA, self.airsim.ImageType.DepthPlanar, True, False)])
        if rs[0].width == 0:
            return None, None
        img = np.frombuffer(rs[0].image_data_uint8, np.uint8).reshape(rs[0].height, rs[0].width, 3).copy()
        d = np.array(rs[1].image_data_float, np.float32).reshape(rs[1].height, rs[1].width)
        self._last_depth = d
        return img, d

    def pose(self):
        p = self.client.simGetVehiclePose()
        yaw = math.degrees(self.airsim.to_eularian_angles(p.orientation)[2])
        return p.position.x_val, p.position.y_val, p.position.z_val, yaw

    def yaw_deg(self):
        return self.pose()[3]

    def vel_body(self):
        """Measured (vx, vy) in the body frame, m/s."""
        st = self.client.getMultirotorState().kinematics_estimated
        yaw = self.airsim.to_eularian_angles(st.orientation)[2]
        v = st.linear_velocity
        c, s = math.cos(yaw), math.sin(yaw)
        return c * v.x_val + s * v.y_val, -s * v.x_val + c * v.y_val

    def alt_m(self):
        """Vehicle height above ground (camera height minus the camera's mount offset)."""
        return self.cam_height_m() - 0.05

    def measure_ground(self):
        """Measure the ground's NED z through the depth image; call while hovering over
        clear ground (take-off does this). Returns the z or None."""
        self.frame_and_depth()
        h = self._depth_ground_height()
        if h is None:
            return None
        self.ground_z_ned = self.pose()[2] + h
        return self.ground_z_ned

    def takeoff(self):
        # a previous run may have left the drone airborne inside a canopy: start clean
        # (vehicle reset only; spawned trees stay where they are)
        self.client.reset()
        time.sleep(0.5)
        self.home_z = self.pose()[2]
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync(timeout_sec=10).join()
        gz = self.measure_ground()
        if gz is None:
            gz = self.home_z + 0.9
            self.ground_z_ned = gz
        self.client.moveToZAsync(gz - self.cfg.cruise_alt_m, 1.0, timeout_sec=12).join()
        print(f"[airsim] ground z {gz:.2f}, camera now {self.cam_height_m():.2f} m above ground")

    def land(self):
        self.client.landAsync(timeout_sec=15).join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def send(self, cmd: Command, dt):
        # Flying into a canopy pins the drone against the collision hull: commands keep
        # coming, nothing moves. Sim-only recovery: if we ask for motion and get none for
        # ~1.5 s, push away along the collision normal. (A Tello would just be crashed.)
        self._stuck_t = getattr(self, "_stuck_t", 0.0)
        self.recoveries = getattr(self, "recoveries", 0)
        want = math.hypot(cmd.vx, cmd.vy)
        if want > 0.2:
            v = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            got = math.hypot(v.x_val, v.y_val)
            self._stuck_t = self._stuck_t + dt if got < 0.08 else 0.0
            if self._stuck_t > 1.5:
                ci = self.client.simGetCollisionInfo()
                n = ci.normal
                nx, ny = (n.x_val, n.y_val) if abs(n.x_val) + abs(n.y_val) > 1e-3 else (-1.0, 0.0)
                k = 0.8 / max(1e-6, math.hypot(nx, ny))
                print(f"[airsim] stuck on {ci.object_name or '?'}: pushing away along the normal")
                self.recoveries += 1
                self._stuck_t = 0.0
                self.client.moveByVelocityAsync(nx * k, ny * k, 0.0, 1.5).join()
                return True
        else:
            self._stuck_t = 0.0
        alt_err = self.cfg.cruise_alt_m - self.alt_m()
        vz = -self.cfg.k_alt * alt_err + cmd.vz           # NED: down positive
        self.client.moveByVelocityBodyFrameAsync(cmd.vx, cmd.vy, vz, dt,
                                                 self.airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                 self.airsim.YawMode(True, cmd.yaw_rate))
        return False

    def close(self):
        pass


class TelloDrone:
    def __init__(self, cfg: OrbitConfig):
        from djitellopy import Tello
        self.cfg = cfg
        self.tello = Tello()
        self.tello.connect()
        self.battery = self.tello.get_battery()
        print(f"[tello] connected, battery {self.battery}%")
        if self.battery < cfg.tello_min_battery:
            raise RuntimeError(f"battery {self.battery}% below {cfg.tello_min_battery}%")
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()
        self.yaw0 = None
        self._last_frame_t = time.time()

    def frame(self):
        f = self.reader.frame
        if f is None:
            return None
        self._last_frame_t = time.time()
        import cv2
        return cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if self.cfg.frame_is_rgb else f.copy()

    def depth(self):
        return None

    def frame_and_depth(self):
        return self.frame(), None

    def yaw_deg(self):
        y = float(self.tello.get_yaw())
        if self.yaw0 is None:
            self.yaw0 = y
        return (y - self.yaw0 + 180.0) % 360.0 - 180.0

    def alt_m(self):
        return self.tello.get_height() / 100.0

    def cam_height_m(self):
        """Downward time-of-flight range (cm), the most direct height-above-ground the Tello has."""
        try:
            tof = self.tello.get_distance_tof() / 100.0
            return tof if 0.2 < tof < 8.0 else self.alt_m()
        except Exception:
            return self.alt_m()

    def pitch_deg(self):
        try:
            return float(self.tello.get_pitch())
        except Exception:
            return 0.0

    def vel_body(self):
        """Tello's own velocity estimate (optical flow + IMU), cm/s -> m/s, body frame."""
        try:
            return self.tello.get_speed_x() / 100.0, self.tello.get_speed_y() / 100.0
        except Exception:
            return None

    def takeoff(self):
        self.tello.takeoff()
        time.sleep(1.0)
        # Tello takes off to ~0.8 m; climb to cruise altitude
        dz = int(round((self.cfg.cruise_alt_m - self.alt_m()) * 100))
        if dz >= 20:
            self.tello.move_up(min(dz, 150))

    def land(self):
        self.tello.send_rc_control(0, 0, 0, 0)
        self.tello.land()

    def send(self, cmd: Command, dt):
        c = self.cfg
        alt_err = c.cruise_alt_m - self.alt_m()
        up = c.k_alt * alt_err - cmd.vz
        rc = lambda v, k: int(max(-100, min(100, round(v * k))))  # noqa: E731
        # djitellopy: left_right (+right), forward_backward (+fwd), up_down (+up), yaw (+clockwise)
        self.tello.send_rc_control(rc(cmd.vy, c.rc_per_mps), rc(cmd.vx, c.rc_per_mps),
                                   rc(up, c.rc_per_mps), rc(cmd.yaw_rate, c.rc_per_dps))

    def hover(self):
        self.tello.send_rc_control(0, 0, 0, 0)

    def close(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.streamoff()
            self.tello.end()
        except Exception:
            pass
