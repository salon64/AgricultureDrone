#!/usr/bin/env python3
"""
teleop_keyboard.py - fly the AirSim drone from the keyboard and look through its camera.

Opens an OpenCV window with the front_center camera (RGB, or depth with C) and reads
keys from that window (click it first). Commands are body-frame velocities, resent
every tick while a key is held.

    T        take off (arms + API control)      L   land
    W / S    forward / back                     A / D   strafe left / right
    Q / E    yaw left / right                   R / F   climb / descend
    Space    hover (zero velocity)              C   toggle RGB / depth view
    1..4     speed 0.5 / 1 / 2 / 4 m/s          P   print pose + nearest trees
    Esc      quit (keeps the drone where it is)

Tips for the Unreal window itself: F = first person, B = chase, / = ground observer,
M = free camera (then arrow keys / mouse move it), 0 = back to the default view.
Run with the explore settings (ClockSpeed 1) or it is hard to control:
    bc_learning/airsim/explore_orchard.bat
"""
import math
import sys
import time

import airsim
import cv2
import numpy as np

SPEEDS = {ord("1"): 0.5, ord("2"): 1.0, ord("3"): 2.0, ord("4"): 4.0}
CMD_DURATION = 0.25        # s, each key tick re-issues the command
YAW_RATE_DEG = 45.0


def tree_list(client):
    out = []
    for name in client.simListSceneObjects("Tree.*"):
        p = client.simGetObjectPose(name).position
        if p.x_val > -100.0:
            out.append((name, p.x_val, p.y_val))
    return out


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    speed, show_depth = 1.0, False
    print(__doc__)
    win = "AirSim teleop (click here, Esc quits)"
    cv2.namedWindow(win)
    trees = tree_list(client)
    print(f"[teleop] {len(trees)} trees in the scene: {[(n, round(x, 1), round(y, 1)) for n, x, y in trees]}")

    while True:
        if show_depth:
            r = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)])[0]
            d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
            img = cv2.applyColorMap(np.clip(d / 25.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        else:
            r = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])[0]
            img = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3).copy()
        img = cv2.resize(img, (r.width * 3, r.height * 3), interpolation=cv2.INTER_NEAREST)

        st = client.getMultirotorState()
        p = st.kinematics_estimated.position
        yaw = math.degrees(airsim.to_eularian_angles(st.kinematics_estimated.orientation)[2])
        hud = f"x={p.x_val:5.1f} y={p.y_val:5.1f} alt={-p.z_val:4.1f}m yaw={yaw:5.0f}  speed {speed:.1f} m/s  [{'depth' if show_depth else 'rgb'}]"
        cv2.putText(img, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)

        k = cv2.waitKey(30) & 0xFF
        vx = vy = vz = 0.0
        yaw_rate = 0.0
        send = False
        if k == 27:
            break
        elif k == ord("t"):
            client.enableApiControl(True); client.armDisarm(True)
            client.takeoffAsync().join()
            client.moveToZAsync(p.z_val - 2.0, 1.0).join()
            print("[teleop] airborne")
        elif k == ord("l"):
            client.landAsync().join(); print("[teleop] landed")
        elif k == ord("c"):
            show_depth = not show_depth
        elif k in SPEEDS:
            speed = SPEEDS[k]; print(f"[teleop] speed {speed} m/s")
        elif k == ord("p"):
            trees = tree_list(client)
            near = sorted(trees, key=lambda t: math.hypot(t[1] - p.x_val, t[2] - p.y_val))[:3]
            print(f"[teleop] pose ({p.x_val:.1f},{p.y_val:.1f},alt {-p.z_val:.1f}) yaw {yaw:.0f}; nearest trees: "
                  f"{[(n, round(math.hypot(x - p.x_val, y - p.y_val), 1)) for n, x, y in near]}")
        elif k == ord("w"): vx, send = speed, True
        elif k == ord("s"): vx, send = -speed, True
        elif k == ord("a"): vy, send = -speed, True
        elif k == ord("d"): vy, send = speed, True
        elif k == ord("r"): vz, send = -speed, True
        elif k == ord("f"): vz, send = speed, True
        elif k == ord("q"): yaw_rate, send = -YAW_RATE_DEG, True
        elif k == ord("e"): yaw_rate, send = YAW_RATE_DEG, True
        elif k == ord(" "): send = True
        if send:
            client.moveByVelocityBodyFrameAsync(vx, vy, vz, CMD_DURATION,
                                                airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(True, yaw_rate))
    cv2.destroyAllWindows()
    print("[teleop] bye (drone left under API control; T/L again any time)")


if __name__ == "__main__":
    sys.exit(main())
