@echo off
REM Rebuilds the Blocks editor target (and the AirSim plugin inside it) with UnrealBuildTool.
REM Needed after patching the AirSim plugin (see airsim_scan_game_assets.patch).
REM Close any running Blocks instance first; UBT uses the VS 2022 toolset pinned in
REM %APPDATA%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml
"C:\Program Files\Epic Games\UE_4.27\Engine\Build\BatchFiles\Build.bat" BlocksEditor Win64 Development -Project="C:\Users\malco\projects\AirSim\Unreal\Environments\Blocks\Blocks.uproject" -WaitMutex
