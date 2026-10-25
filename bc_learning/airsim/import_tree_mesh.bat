@echo off
REM Imports bc_learning/airsim/assets/Tree_LowPoly.obj into the Blocks project as /Game/Orchard/Tree_LowPoly
REM Run from any shell (full path). Takes 1-3 minutes the first time (shader compile).
set UE=C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor-Cmd.exe
set PROJ=C:\Users\malco\projects\AirSim\Unreal\Environments\Blocks\Blocks.uproject
set JSON=C:\Users\malco\projects\AgricultureDrone\bc_learning\airsim\import_tree_mesh.json
"%UE%" "%PROJ%" -run=ImportAssets -importsettings="%JSON%" -replaceexisting -nosourcecontrol -unattended -nopause -nosplash -stdout -FullStdOutLogOutput
