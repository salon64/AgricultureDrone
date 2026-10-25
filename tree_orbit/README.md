# tree_orbit: detect a tree, orbit it, move on

A modular, non-learned controller for the orchard inspection task. It lives next to
the RL code in `bc_learning/` and does not change it. The same controller runs in
AirSim and on a DJI Tello; only the drone backend differs.

```
camera frame ──► detector.py ──► controller.py ──► drones.py ──► AirSim | Tello
                 bearing+range    SEARCH/APPROACH    body-frame
                 of each tree     ORBIT/LEAVE        velocities
                                      ▲
                                  odometry.py  (dead reckoning + "trees I finished")
```

## Why this instead of the end-to-end policy

The RL policy in `bc_learning/scripts` learns perception, memory and control from
pixels in one network and needs a depth image. The Tello has no depth camera, no
position sensor and a cheap RGB stream. This pipeline needs only a tree bearing and a
rough range per frame, which an RGB detector can give, and it is testable piece by
piece. It also scores against the old environment's reward, so RL and this baseline
can be compared with one number.

## Files

| File | Role |
|---|---|
| `config.py` | all tunables, `OrbitConfig.sim()` and `OrbitConfig.tello()` presets |
| `detector.py` | `GreenBlobDetector` (RGB, runs on the Tello), `DepthBlobDetector` (sim), `draw()` overlay |
| `odometry.py` | dead reckoning from commanded velocity + IMU yaw, memory of finished trees |
| `controller.py` | `Mission` state machine, outputs `Command(vx, vy, vz, yaw_rate)` |
| `drones.py` | `AirSimDrone`, `TelloDrone` behind the same 6-method interface |
| `run_sim.py` | fly in the Blocks orchard map; `--scored` drives the old `OrchardEnv` for a fair score |
| `run_tello.py` | fly the real Tello with a video window and keyboard abort |
| `orbit_rl_env.py` | compact-observation gym env (wraps `OrchardEnv`) with wind + noise randomisation |
| `train_orbit_rl.py` / `eval_wind.py` | PPO training on it, and a wind sweep scoring both controllers |
| `calibrate_sim.py` | fits the canopy-bottom height the RGB range estimate needs |

## How the controller works

1. **Detect.** Canopy green is segmented in HSV, blobs become detections. Bearing
   comes from the blob centre and the camera's horizontal field of view. Range comes
   from the blob width and the known crown diameter (3.2 m sim tree, about 3 m for a
   mature mikan). If a depth image exists it is used instead.
2. **Search.** Rotate in place until a detection is seen that is not near a finished tree.
3. **Approach.** Yaw rate proportional to bearing keeps the tree centred, forward speed
   proportional to range error drives to the orbit ring (4 m in sim, 3 m on the Tello).
4. **Orbit.** Strafe sideways at constant speed while still keeping the tree centred.
   Because the heading must turn with the orbit, the yaw accumulated from the IMU equals
   the orbit angle. After 380 degrees the tree is done, its position is stored in the
   odometry frame. If another tree sits closer than 2.6 m on the side the drone is
   strafing towards, the orbit direction flips; with 5 m between trunks the 3.2 m ring
   only just clears the neighbours' canopies.
5. **Leave.** Back off and turn away so the finished tree leaves the frame, then search.
   Detections that project within 3 m of a finished tree are ignored.

Losing the target for a second keeps the last command, four seconds aborts to search.

## Running in the simulator

```
bc_learning\airsim\launch_orchard.bat explore          (real-time clock, for watching)
cd bc_learning\scripts && python place_layout.py --layout kinpo_kawachi_east --n 6
cd ..\..\tree_orbit && python run_sim.py --show --max-trees 3
```

Scored comparison with the RL reward (ClockSpeed 5, same env, same penalties):

```
bc_learning\airsim\launch_orchard.bat train
python run_sim.py --scored --episodes 5 --detector green
python run_sim.py --scored --episodes 5 --detector depth
```

It prints reward, sectors collected, trees finished by the env's own bookkeeping and by
the controller's, and the reason the episode ended. Run the RL policy with
`bc_learning/scripts/run_rl.py` on the same layouts for the other side of the table.

## Learned controller on the same perception

`orbit_rl_env.py` wraps the unchanged `OrchardEnv` but gives the policy a 16-number
observation instead of pixels: target bearing and range, a second tree's bearing and
range, measured body velocity and yaw rate, sectors covered on the current tree, a
"near a finished tree" flag and the last action. Every one of those is something the
Tello can measure, so the policy transfers. Wind (random heading, 0 to `--wind-max`
m/s, with gusts), detector noise, the layout window and tree scale are randomised
every episode.

```
bc_learningirsim\launch_orchard.bat train
python train_orbit_rl.py --timesteps 300000 --wind-max 3.0     (PPO, 2x128 MLP, CPU)
python eval_wind.py --controller both --winds 0 2 4 --episodes 3
```

`eval_wind.py` scores the feedback controller and the trained policy with the same
reward under the same wind levels, which is the comparison that decides what flies.

## Range calibration

`calibrate_sim.py` stands at known ranges from a tree and solves `canopy_bottom_m`
(the height of the lowest canopy pixel the mask keeps) from ground truth. Rerun it
after changing the tree mesh, lighting or HSV thresholds. On a real tree, measure the
lowest leaves with a tape and set it in `OrbitConfig.tello()`.

## Running on the Tello

Everything is in `bc_env` (`pip install djitellopy --no-deps`, the package otherwise
drags in a NumPy and OpenCV that break the pinned stack).

1. Connect to the Tello Wi-Fi. `python run_tello.py --dry` shows the camera, the
   detections and (H key) the HSV mask with the drone in your hand. Point it at a tree.
   If the canopy is not boxed, adjust `hsv_lo` / `hsv_hi` in `OrbitConfig.tello()`.
2. Calibrate speed once: `rc_per_mps` maps m/s to Tello rc units. The default 100 is
   roughly right indoors; outdoors with wind it is optimistic.
3. `python run_tello.py --max-trees 1` takes off, climbs to 1.5 m, runs the mission.
   Esc or L lands immediately, Space pauses in a hover. Video loss for 2 s lands.
4. Fly where the tree has 3 m of clear space all around and no other green at tree
   height in the background.

### Known limits on the Tello

- The green detector fails when the background is also green at canopy height
  (hedge rows, dense orchards). A learned detector (small YOLO on the sim frames plus a
  few hundred real ones) fits the same `detect()` interface.
- Range from blob width is only good to about 20 percent. The orbit tolerates that;
  the radius just drifts.
- Dead reckoning drifts with wind. It only needs to hold for the minute or two it takes
  to finish a few trees.
- Terraced mikan orchards: the Tello holds height above take-off with its barometer and
  downward sensor, so on a slope the drone may dip into the canopy of the downhill
  row. Keep `cruise_alt_m` above canopy height for the first flights.
