#!/usr/bin/env python3
"""
make_layout.py - build orchard layout CSVs for orchard_env.py (see orchard_layouts.py)

Examples
--------
# 1. Real orchard from open aerial imagery (GSI tiles for Japan), 60 m around a point:
python make_layout.py aerial --lat 32.6421 --lon 130.5873 --radius 60 \
        --crown-r 2.0 --min-spacing 3.5 --name uki_mikan_01 --crop mikan
#    -> layouts/uki_mikan_01.csv + layouts/uki_mikan_01_preview.png  (check the circles!)
#    Optionally restrict to a boundary polygon traced in Google Earth: --roi-kml field.kml

# 2. From a Google Earth KML (File > Save Place As...). Placemarks = trees,
#    paths = rows (trees every --in-row metres), polygons = blocks (row fill):
python make_layout.py kml --file my_orchard.kml --in-row 4.0 --row 6.0 --name site_a

# 3. Synthetic grid in the same format (for curriculum / sanity checks):
python make_layout.py grid --rows 3 --cols 6 --row 6.0 --in-row 5.0 --name grid_3x6

# 4. Inspect any layout:
python make_layout.py show layouts/uki_mikan_01.csv
"""
import argparse
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchard_layouts as ol  # noqa: E402


def _common_meta(args, **extra):
    m = {"name": args.name}
    for k in ("crop", "note"):
        v = getattr(args, k, None)
        if v:
            m[k] = v.replace(" ", "_")
    m.update(extra)
    return m


def _finish(args, trees, meta, gi=None, poly=None):
    out = os.path.join(args.out_dir, args.name + ".csv")
    ol.write_layout(out, trees, meta)
    print(f"[layout] wrote {out}")
    print(f"[layout] {ol.layout_summary(trees)}")
    if gi is not None and all("lat" in t for t in trees):
        p = ol.render_preview(gi, trees, os.path.join(args.out_dir, args.name + "_preview.png"),
                              poly_latlon=poly, label=args.name)
        print(f"[layout] preview {p}")
    p = ol.render_layout_plan(trees, os.path.join(args.out_dir, args.name + "_plan.png"), title=args.name)
    print(f"[layout] plan    {p}")


def cmd_aerial(args):
    gi = ol.fetch_aerial(args.lat, args.lon, args.radius, source=args.source, zoom=args.zoom)
    print(f"[aerial] {args.source} z{gi.z}  {gi.img.shape[1]}x{gi.img.shape[0]} px  {gi.mpp:.2f} m/px")
    poly = None
    if args.roi_kml:
        k = ol.parse_kml(args.roi_kml)
        if not k["polygons"]:
            sys.exit("ROI KML has no polygon")
        poly = k["polygons"][0]
    dets, thr = ol.detect_crowns(gi, crown_r_m=args.crown_r, min_spacing_m=args.min_spacing,
                                 veg_thresh=args.veg_thresh, roi_poly_latlon=poly)
    # keep only detections inside the requested radius (the mosaic is tile-aligned, so larger)
    dets = [d for d in dets
            if math.hypot(*ol.latlon_to_local(d["lat"], d["lon"], args.lat, args.lon)) <= args.radius]
    if args.max_trees:
        dets = dets[:args.max_trees]
    print(f"[aerial] {len(dets)} crowns (veg index threshold {thr:.3f})")
    trees = []
    for d in dets:
        n, e = ol.latlon_to_local(d["lat"], d["lon"], args.lat, args.lon)
        trees.append({"north_m": n, "east_m": e, "lat": d["lat"], "lon": d["lon"],
                      "crown_r_m": d["crown_r_m"], "scale": 1.0, "yaw_deg": 0.0})
    rs = ol.row_structure([(t["north_m"], t["east_m"]) for t in trees])
    meta = _common_meta(args, origin_lat=f"{args.lat:.7f}", origin_lon=f"{args.lon:.7f}",
                        source=f"{args.source}-aerial-z{gi.z}", radius_m=args.radius,
                        veg_thresh=f"{thr:.3f}")
    if rs:
        meta.update(row_heading_deg=f"{rs['row_heading_deg']:.1f}", in_row_m=f"{rs['in_row_m']:.2f}",
                    row_m=f"{rs['row_m']:.2f}")
    _finish(args, trees, meta, gi=ol.crop_geoimage(gi, args.lat, args.lon, args.radius + 12.0), poly=poly)


def cmd_kml(args):
    k = ol.parse_kml(args.file)
    all_ll = [(la, lo) for la, lo, _ in k["points"]] + [p for l in k["lines"] for p in l] \
        + [p for pg in k["polygons"] for p in pg]
    if not all_ll:
        sys.exit("KML has no Placemark geometry")
    lat0 = float(np.mean([p[0] for p in all_ll])); lon0 = float(np.mean([p[1] for p in all_ll]))
    trees = []
    for la, lo, name in k["points"]:
        n, e = ol.latlon_to_local(la, lo, lat0, lon0)
        trees.append({"north_m": n, "east_m": e, "lat": la, "lon": lo, "crown_r_m": args.crown_r})
    for line in k["lines"]:
        line_ne = [ol.latlon_to_local(la, lo, lat0, lon0) for la, lo in line]
        for n, e in ol.trees_along_line(line_ne, args.in_row):
            la, lo = ol.local_to_latlon(n, e, lat0, lon0)
            trees.append({"north_m": n, "east_m": e, "lat": la, "lon": lo, "crown_r_m": args.crown_r})
    heading = None
    for pg in k["polygons"]:
        poly_ne = [ol.latlon_to_local(la, lo, lat0, lon0) for la, lo in pg]
        pts, heading = ol.fill_polygon_rows(poly_ne, args.row, args.in_row, heading_deg=args.heading)
        for n, e in pts:
            la, lo = ol.local_to_latlon(n, e, lat0, lon0)
            trees.append({"north_m": n, "east_m": e, "lat": la, "lon": lo, "crown_r_m": args.crown_r})
    print(f"[kml] {len(k['points'])} points, {len(k['lines'])} paths, {len(k['polygons'])} polygons "
          f"-> {len(trees)} trees")
    meta = _common_meta(args, origin_lat=f"{lat0:.7f}", origin_lon=f"{lon0:.7f}",
                        source="kml:" + os.path.basename(args.file).replace(" ", "_"),
                        in_row_m=args.in_row, row_m=args.row)
    if heading is not None:
        meta["row_heading_deg"] = f"{heading:.1f}"
    gi = None
    if args.preview:
        r = max(40.0, max(abs(t["north_m"]) for t in trees) + 15, max(abs(t["east_m"]) for t in trees) + 15)
        gi = ol.fetch_aerial(lat0, lon0, r, source=args.source)
    _finish(args, trees, meta, gi=gi, poly=k["polygons"][0] if k["polygons"] else None)


def cmd_grid(args):
    rng = random.Random(args.seed)
    trees = []
    for r in range(args.rows):
        for c in range(args.cols):
            trees.append({"north_m": c * args.in_row + rng.uniform(-args.jitter, args.jitter),
                          "east_m": (r - (args.rows - 1) / 2.0) * args.row + rng.uniform(-args.jitter, args.jitter),
                          "crown_r_m": args.crown_r})
    meta = _common_meta(args, source="synthetic-grid", in_row_m=args.in_row, row_m=args.row,
                        row_heading_deg=0.0)
    _finish(args, trees, meta)


def cmd_show(args):
    for p in args.files:
        trees, meta = ol.read_layout(p)
        print(f"{p}: {ol.layout_summary(trees)}")
        print("   meta:", meta)
        if args.plan:
            out = os.path.splitext(p)[0] + "_plan.png"
            ol.render_layout_plan(trees, out, title=os.path.basename(p))
            print("   plan:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=ol.LAYOUT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aerial", help="detect crowns in open ortho imagery")
    a.add_argument("--lat", type=float, required=True); a.add_argument("--lon", type=float, required=True)
    a.add_argument("--radius", type=float, default=60.0, help="half-size of the square to analyse (m)")
    a.add_argument("--source", choices=list(ol.TILE_SOURCES), default="gsi")
    a.add_argument("--zoom", type=int, default=None)
    a.add_argument("--crown-r", type=float, default=2.0, help="typical crown radius (m)")
    a.add_argument("--min-spacing", type=float, default=3.5, help="min distance between detections (m)")
    a.add_argument("--veg-thresh", type=float, default=None, help="excess-green threshold (-1..1); default Otsu")
    a.add_argument("--roi-kml", default=None, help="KML polygon: only detect inside it")
    a.add_argument("--max-trees", type=int, default=0)
    a.add_argument("--name", required=True); a.add_argument("--crop", default=""); a.add_argument("--note", default="")
    a.set_defaults(fn=cmd_aerial)

    k = sub.add_parser("kml", help="trees from Google Earth placemarks / paths / polygons")
    k.add_argument("--file", required=True)
    k.add_argument("--in-row", type=float, default=4.0); k.add_argument("--row", type=float, default=6.0)
    k.add_argument("--heading", type=float, default=None, help="row heading for polygons (deg from north)")
    k.add_argument("--crown-r", type=float, default=2.0)
    k.add_argument("--preview", action="store_true", help="also fetch imagery and draw a preview")
    k.add_argument("--source", choices=list(ol.TILE_SOURCES), default="gsi")
    k.add_argument("--name", required=True); k.add_argument("--crop", default=""); k.add_argument("--note", default="")
    k.set_defaults(fn=cmd_kml)

    g = sub.add_parser("grid", help="synthetic rows")
    g.add_argument("--rows", type=int, default=2); g.add_argument("--cols", type=int, default=4)
    g.add_argument("--row", type=float, default=8.0); g.add_argument("--in-row", type=float, default=8.0)
    g.add_argument("--jitter", type=float, default=0.5); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--crown-r", type=float, default=2.0)
    g.add_argument("--name", required=True); g.add_argument("--crop", default=""); g.add_argument("--note", default="")
    g.set_defaults(fn=cmd_grid)

    s = sub.add_parser("show", help="summarise layout CSVs")
    s.add_argument("files", nargs="+"); s.add_argument("--plan", action="store_true")
    s.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
