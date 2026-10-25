@echo off
REM Builds /Game/Orchard/OrchardMap (Blocks map without the stock props). Close Blocks first.
REM Uses the PythonScript commandlet (no editor window). Takes ~1-2 min.
set UE=C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor-Cmd.exe
set PROJ=C:\Users\malco\projects\AirSim\Unreal\Environments\Blocks\Blocks.uproject
set SCRIPT=C:\Users\malco\projects\AgricultureDrone\bc_learning\airsim\make_orchard_map.py
"%UE%" "%PROJ%" -run=pythonscript -script="%SCRIPT%" -unattended -nopause -nosplash -stdout -FullStdOutLogOutput
