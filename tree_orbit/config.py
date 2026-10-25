"""Tunables for the tree-orbit pipeline. One dataclass, two presets (sim / Tello)."""
from dataclasses import dataclass, field


@dataclass
class OrbitConfig:
    # --- camera ---------------------------------------------------------------
    hfov_deg: float = 90.0          # horizontal field of view (AirSim front_center = 90)
    frame_w: int = 320
    frame_h: int = 256

    # --- what a tree looks like ----------------------------------------------
    crown_diameter_m: float = 3.2   # fallback: canopy pixel width -> range (RGB detector)
    canopy_bottom_m: float = 1.42   # height of the lowest canopy pixel the mask keeps (calibrate_sim.py:
                                    # the unlit underside of the sim canopy drops out, so 1.3 not 0.5)
                                    # height this turns the blob's bottom row into range (merge-proof)
    hsv_lo: tuple = (40, 30, 25)    # canopy green, HSV lower bound (OpenCV ranges H 0-179);
    hsv_hi: tuple = (95, 255, 255)  # loose on S/V so the sunlit side of the canopy stays in the blob
    min_blob_frac: float = 0.0015   # min blob area as a fraction of the frame
    max_range_m: float = 14.0       # ignore trees further than this (depth detector / sanity)

    # --- orbit geometry ----------------------------------------------------------
    orbit_radius_m: float = 3.2     # distance to hold from the trunk (OrchardEnv wants 3.0-5.5;
                                    # neighbours 5 m away leave no room for a wider ring)
    radius_tol_m: float = 0.6       # "on the ring" tolerance before orbiting starts
    orbit_speed_mps: float = 0.5    # tangential speed while orbiting
    orbit_total_deg: float = 380.0  # yaw to accumulate before a tree counts as done
    approach_speed_mps: float = 0.8
    min_safe_dist_m: float = 2.2    # back off hard inside this
    obstacle_dist_m: float = 2.6    # a non-target tree closer than this on the strafe side flips the orbit
    flip_cooldown_s: float = 3.0
    cruise_alt_m: float = 2.8       # camera height above ground to hold (what the RL env's 2.5 m above
                                    # spawn works out to); higher camera = better ground-plane range

    # --- control gains ---------------------------------------------------------
    k_yaw: float = 0.9              # deg/s per degree of bearing error
    yaw_deadband_deg: float = 3.0   # ignore bearing errors smaller than this
    k_dist: float = 0.7             # m/s per metre of distance error
    max_yaw_rate_dps: float = 45.0
    max_speed_mps: float = 1.0
    max_accel_mps2: float = 0.5     # slew limit on commanded velocity (removes the lurches)
    max_yaw_accel_dps2: float = 60.0
    k_alt: float = 0.8              # m/s per metre of altitude error (Tello / unscored sim)

    # --- behaviour timing ----------------------------------------------------------
    search_yaw_rate_dps: float = 18.0
    lost_grace_s: float = 1.0       # keep last command this long after losing the target
    lost_abort_s: float = 4.0       # then give up and search again
    leave_s: float = 8.0            # max time to back out / turn away after finishing a tree
    leave_clear_deg: float = 75.0   # leave ends once the finished tree is this far off the nose
    done_tree_radius_m: float = 3.0 # a detection this close to a finished tree is ignored
    bearing_gate_deg: float = 20.0  # target association gate frame to frame
    max_trees: int = 0              # 0 = unlimited
    mission_timeout_s: float = 600.0

    # --- Tello only ---------------------------------------------------------------
    rc_per_mps: float = 100.0       # rc units per m/s (Tello: 100 rc ~ 1 m/s, calibrate)
    rc_per_dps: float = 1.0         # rc units per deg/s of yaw
    tello_min_battery: int = 25
    frame_is_rgb: bool = True       # djitellopy >= 2.5 returns RGB frames

    extra: dict = field(default_factory=dict)

    @classmethod
    def sim(cls, **kw):
        return cls(**kw)

    @classmethod
    def tello(cls, **kw):
        """DJI Tello: 960x720 camera, ~82.6 deg diagonal FOV -> ~69.5 deg horizontal.
        Smaller radius and speeds for a real orchard; crown diameter of a mature mikan ~3 m."""
        d = dict(hfov_deg=69.5, frame_w=960, frame_h=720, crown_diameter_m=3.0, canopy_bottom_m=0.4,
                 orbit_radius_m=3.0, orbit_speed_mps=0.35, approach_speed_mps=0.4,
                 max_speed_mps=0.5, max_yaw_rate_dps=35.0, cruise_alt_m=1.5,
                 min_safe_dist_m=1.8, search_yaw_rate_dps=15.0, mission_timeout_s=240.0)
        d.update(kw)
        return cls(**d)
