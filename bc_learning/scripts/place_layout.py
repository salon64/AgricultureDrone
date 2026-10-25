#!/usr/bin/env python3
"""
place_layout.py - put a real orchard layout into the running AirSim map without flying.

Uses the same code path as OrchardEnv.reset() (tree pool + layout window) so what you
see is exactly what training sees, but leaves the drone on the ground for exploring.

    python place_layout.py                       # random layout, 4-8 trees, random heading
    python place_layout.py --layout kinpo_kawachi_block --n 8 --rotate 0
    python place_layout.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchard_env as oe  # noqa: E402
import orchard_layouts as ol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=None, help="layout name (without .csv) or path; default random")
    ap.add_argument("--n", type=int, default=None, help="trees to place (default LAYOUT_TREES_MIN..MAX)")
    ap.add_argument("--rotate", type=float, default=None, help="window heading in deg (default random)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for f in ol.list_layouts():
            trees, meta = ol.read_layout(f)
            print(f"{os.path.basename(f)[:-4]:24s} {ol.layout_summary(trees)}")
        return
    if a.n is not None:
        oe.LAYOUT_TREES_MIN = oe.LAYOUT_TREES_MAX = a.n
        oe.TREE_POOL_SIZE = max(oe.TREE_POOL_SIZE, a.n)
    if a.rotate is not None:
        oe.LAYOUT_ROTATE_DEG = (a.rotate, a.rotate)
    oe.LAYOUT_MODE = "file"

    env = oe.OrchardEnv()
    if a.layout:
        path = a.layout if os.path.isfile(a.layout) else os.path.join(ol.LAYOUT_DIR, a.layout + ".csv")
        if not os.path.isfile(path):
            sys.exit(f"no such layout: {path}")
        env._layout_files = [path]
    env.client.reset()
    env._ensure_tree_pool()
    env._randomize_trees()
    trees = oe._get_tree_positions(env.client)
    print(f"[place] {len(trees)} trees in front of the drone (x forward, y right):")
    for name, x, y, z in trees:
        print(f"   {name:8s} x={x:6.1f}  y={y:6.1f}")
    print("[place] now run teleop_keyboard.py to fly, or train_rl.py to train")


if __name__ == "__main__":
    main()
