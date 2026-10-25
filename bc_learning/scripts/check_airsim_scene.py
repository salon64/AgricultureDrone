#!/usr/bin/env python3
"""
check_airsim_scene.py — run this with the Unreal/AirSim map already playing.
Confirms what orchard_env.py needs before you try training:
  1. RPC connection works
  2. a 'front_center' camera exists and reports its FOV
  3. Scene + DepthPlanar images come back at a sane resolution
  4. actors matching 'Tree.*' exist, are listed, and can be moved (Mobility=Movable)
"""
import airsim
import numpy as np

from orchard_env import CAMERA_HFOV_DEG, CLOCK_SPEED

client = airsim.MultirotorClient()
client.confirmConnection()
print("[1] connected")

info = client.simGetCameraInfo("front_center")
print(f"[2] front_center FOV = {info.fov:.1f} deg  (env expects {CAMERA_HFOV_DEG:.1f}; "
      f"env ClockSpeed constant = {CLOCK_SPEED}, must match settings.json)")

rs = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
])
scene, depth = rs
print(f"[3] Scene {scene.width}x{scene.height}  DepthPlanar {depth.width}x{depth.height}  "
      f"depth min/max = {min(depth.image_data_float):.2f}/{max(depth.image_data_float):.2f} m")

from orchard_env import LAYOUT_MODE, TREE_ASSET_NAME, TREE_POOL_SIZE

names = []
for pattern in ("Tree.*", ".*[Tt]ree.*"):
    names = client.simListSceneObjects(pattern)
    if names:
        break
print(f"[4] tree actors found: {len(names)} -> {names}")
assets = client.simListAssets()
has_asset = TREE_ASSET_NAME in assets
print(f"    spawnable asset '{TREE_ASSET_NAME}' in registry: {has_asset}  "
      f"(LAYOUT_MODE={LAYOUT_MODE}, TREE_POOL_SIZE={TREE_POOL_SIZE})")
if TREE_POOL_SIZE > 0 and len(names) < TREE_POOL_SIZE and not has_asset:
    raise SystemExit("    FAIL: the env would need to spawn trees but the mesh asset is missing. "
                     "    Run bc_learning/airsim/import_tree_mesh.bat, make sure the AirSim plugin has "
                     "    airsim_scan_game_assets.patch applied (rebuild_blocks_editor.bat), restart the map.")
if not names and has_asset:
    pose = airsim.Pose(airsim.Vector3r(-200.0, 0.0, client.simGetVehiclePose().position.z_val),
                       airsim.to_quaternion(0, 0, 0))
    spawned = client.simSpawnObject("TreeP00", TREE_ASSET_NAME, pose, airsim.Vector3r(1, 1, 1), False, False)
    names = client.simListSceneObjects("Tree.*")
    print(f"    spawned test tree '{spawned}' (parked at x=-200) -> Tree.* now {names}")
if not names:
    raise SystemExit("    FAIL: no actors whose *internal* name matches Tree.* and nothing to spawn — "
                     "rename them in the editor (F2 in World Outliner) and re-check.")

name = names[0]
pose = client.simGetObjectPose(name)
test = airsim.Pose(airsim.Vector3r(pose.position.x_val + 0.5, pose.position.y_val, pose.position.z_val),
                   pose.orientation)
ok = client.simSetObjectPose(name, test, teleport=True)
after = client.simGetObjectPose(name)
moved = abs(after.position.x_val - test.position.x_val) < 1e-2
client.simSetObjectPose(name, pose, teleport=True)  # put it back
print(f"    simSetObjectPose on {name}: api_ok={ok} moved={moved}")
if not moved:
    raise SystemExit("    FAIL: actor did not move — set its Mobility to Movable in the Details panel.")
scale = client.simGetObjectScale(name)
print(f"    scale = ({scale.x_val:.2f}, {scale.y_val:.2f}, {scale.z_val:.2f})")
print("\nAll checks passed — orchard_env.py should be able to run against this map.")
