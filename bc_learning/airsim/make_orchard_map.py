"""
make_orchard_map.py - Unreal *editor* Python script (not run from bc_env).

Creates /Game/Orchard/OrchardMap from the stock Blocks FlyingExampleMap by deleting
every static-mesh prop (cubes, cylinders, cone, orange ball) and keeping the ground,
lights, sky, fog and player start. Lights are made Movable so the map never asks
for a lighting rebuild.

Run headless (Blocks must be closed):
  "C:\\Program Files\\Epic Games\\UE_4.27\\Engine\\Binaries\\Win64\\UE4Editor-Cmd.exe" ^
      "C:\\Users\\malco\\projects\\AirSim\\Unreal\\Environments\\Blocks\\Blocks.uproject" ^
      -ExecutePythonScript="C:\\Users\\malco\\projects\\AgricultureDrone\\bc_learning\\airsim\\make_orchard_map.py"
(see make_orchard_map.bat). Needs the PythonScriptPlugin enabled in Blocks.uproject.
"""
import unreal

SRC = "/Game/FlyingCPP/Maps/FlyingExampleMap"
DST = "/Game/Orchard/OrchardMap"
KEEP_MESH_PREFIXES = ("Ground",)

ell = unreal.EditorLevelLibrary
log = unreal.log

log(f"[orchard] loading {SRC}")
if not ell.load_level(SRC):
    raise RuntimeError("could not load source map")

world = ell.get_editor_world()
log(f"[orchard] saving copy as {DST}")
if not unreal.EditorLoadingAndSavingUtils.save_map(world, DST):
    raise RuntimeError("save_map failed")

# re-open the copy so that edits go to the new package only
if not ell.load_level(DST):
    raise RuntimeError("could not load the new map")

deleted, kept, movable = [], [], []
for actor in list(ell.get_all_level_actors()):
    name = actor.get_name()
    if isinstance(actor, unreal.StaticMeshActor):
        if name.startswith(KEEP_MESH_PREFIXES):
            kept.append(name)
            continue
        deleted.append(name)
        ell.destroy_actor(actor)
    elif isinstance(actor, unreal.Light):
        comp = actor.get_component_by_class(unreal.LightComponent)
        if comp is not None:
            comp.set_mobility(unreal.ComponentMobility.MOVABLE)
            movable.append(name)
    elif isinstance(actor, unreal.SkyLight):
        comp = actor.get_component_by_class(unreal.SkyLightComponent)
        if comp is not None:
            comp.set_mobility(unreal.ComponentMobility.MOVABLE)
            movable.append(name)

log(f"[orchard] deleted {len(deleted)} props, kept meshes {kept}, movable lights {movable}")
if not ell.save_current_level():
    raise RuntimeError("save_current_level failed")
log("[orchard] done")
