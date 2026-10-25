#!/usr/bin/env python3
"""wait_airsim.py - block until the AirSim RPC port answers (used by explore_orchard.bat)."""
import socket
import sys
import time

timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
t0 = time.time()
while time.time() - t0 < timeout:
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", 41451))
        s.close()
        time.sleep(4)
        print(f"AirSim up after {time.time() - t0:.0f} s")
        sys.exit(0)
    except OSError:
        time.sleep(3)
sys.exit("AirSim did not come up")
