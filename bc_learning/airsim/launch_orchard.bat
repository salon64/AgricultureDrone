@echo off
REM Launches the Blocks/AirSim orchard map in one window.
REM   launch_orchard.bat explore   -> real-time clock (ClockSpeed 1), for flying by hand
REM   launch_orchard.bat train     -> settings.json (ClockSpeed 5), for train_rl.py / run_rl.py
REM   launch_orchard.bat scored    -> settings_scored.json (ClockSpeed 2.5), for tree_orbit eval/training
set MODE=%1
if "%MODE%"=="" set MODE=explore
set UE=C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe
set PROJ=C:\Users\malco\projects\AirSim\Unreal\Environments\Blocks\Blocks.uproject
set HERE=%~dp0
set SETTINGS=%HERE%settings_explore.json
if "%MODE%"=="train" set SETTINGS=%HERE%settings.json
if "%MODE%"=="scored" set SETTINGS=%HERE%settings_scored.json
echo Launching Blocks (%MODE%) with %SETTINGS%
start "AirSim Orchard" "%UE%" "%PROJ%" -game -windowed -ResX=1280 -ResY=720 -NoSound -settings="%SETTINGS%"
