"""
tree_orbit - modular "detect a tree, orbit it, move on" pipeline.

    camera frame -> detector.py  (tree bearing + distance)
                 -> controller.py (state machine: SEARCH / APPROACH / ORBIT / LEAVE)
                 -> drones.py     (AirSim or Tello backend, same body-frame velocity command)

No learning in the loop. Same code runs in the simulator (run_sim.py, scored with the
old OrchardEnv reward so it can be compared to the RL policy) and on a DJI Tello
(run_tello.py). See README.md.
"""
from .config import OrbitConfig
from .detector import Detection, GreenBlobDetector, DepthBlobDetector
from .controller import Command, Mission, State
from .odometry import DeadReckoning

__all__ = ["OrbitConfig", "Detection", "GreenBlobDetector", "DepthBlobDetector",
           "Command", "Mission", "State", "DeadReckoning"]
