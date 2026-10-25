@echo off
REM One-shot: launch the orchard map, place a real layout, open the keyboard teleop window.
REM Extra arguments go to place_layout.py, e.g.  explore_orchard.bat --layout kinpo_kawachi_block --n 8
set HERE=%~dp0
set PY=C:\Users\malco\projects\AgricultureDrone\bc_env\Scripts\python.exe
call "%HERE%launch_orchard.bat" explore
cd /d "%HERE%..\scripts"
echo Waiting for AirSim...
"%PY%" wait_airsim.py 300
if errorlevel 1 exit /b 1
"%PY%" place_layout.py %*
"%PY%" teleop_keyboard.py
