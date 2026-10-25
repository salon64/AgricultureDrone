#!/usr/bin/env python3
"""
orchard_layouts.py - real-world orchard layouts for orchard_env.py

A *layout* is a CSV of tree positions in a local metric frame:

    # origin_lat=32.6421 origin_lon=130.5873 source=gsi-aerial crop=mikan ...
    north_m,east_m,z_m,yaw_deg,scale,crown_r_m,lat,lon
    12.4,-3.1,0.0,0.0,1.0,2.1,32.64221,130.58727

north/east are metres from the origin (north = AirSim +x, east = AirSim +y).
z_m is "down" (AirSim NED) and is 0 for flat layouts. Only the *relative*
positions matter to the environment - it re-origins and may rotate a window
of trees for every episode.

Three ways to build one:
  * from aerial imagery  - detect tree crowns in open ortho tiles
                           (GSI seamlessphoto for Japan, ESRI World Imagery elsewhere)
  * from a KML file      - trace trees (placemarks), rows (paths) or the orchard
                           boundary (polygon) in Google Earth, File > Export
  * from a polygon       - fill a boundary with rows at a given spacing

Everything here is dependency-light: numpy + opencv (already in bc_env).
"""
import csv
import glob
import io
import math
import os
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

import cv2
import numpy as np

LAYOUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "layouts"))
TILE_CACHE = os.path.join(LAYOUT_DIR, ".tile_cache")
EARTH_R = 6378137.0

TILE_SOURCES = {
    # Geospatial Information Authority of Japan, seamless ortho photos. Max zoom 18
    # (~0.5 m/px at 33N). Licence: GSI Tiles terms (attribution "国土地理院").
    "gsi":  ("https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg", 18),
    # Esri World Imagery. Zoom 19 in some areas. Attribution required.
    "esri": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
             "MapServer/tile/{z}/{y}/{x}", 19),
}

KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


# ─────────────────────────────────────────────────────────────────────────────
#  Geodesy helpers
# ─────────────────────────────────────────────────────────────────────────────
def latlon_to_local(lat, lon, lat0, lon0):
    """Equirectangular lat/lon -> (north_m, east_m). Good to <0.1% within a few km."""
    north = math.radians(lat - lat0) * EARTH_R
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return north, east


def local_to_latlon(north, east, lat0, lon0):
    lat = lat0 + math.degrees(north / EARTH_R)
    lon = lon0 + math.degrees(east / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def metres_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_latlon(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
#  Aerial imagery
# ─────────────────────────────────────────────────────────────────────────────
class GeoImage:
    """RGB image + the mapping between pixels and lat/lon (web-mercator tiles)."""

    def __init__(self, img, z, x0, y0, source):
        self.img, self.z, self.x0, self.y0, self.source = img, z, x0, y0, source

    def pix_to_latlon(self, px, py):
        return _tile_latlon(self.x0 + px / 256.0, self.y0 + py / 256.0, self.z)

    def latlon_to_pix(self, lat, lon):
        tx, ty = _tile_xy(lat, lon, self.z)
        return (tx - self.x0) * 256.0, (ty - self.y0) * 256.0

    @property
    def centre_latlon(self):
        h, w = self.img.shape[:2]
        return self.pix_to_latlon(w / 2.0, h / 2.0)

    @property
    def mpp(self):
        return metres_per_pixel(self.centre_latlon[0], self.z)


def _fetch_tile(source, z, x, y, cache_dir=TILE_CACHE, retries=3):
    url_tpl, _ = TILE_SOURCES[source]
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, f"{source}_{z}_{x}_{y}.jpg")
    if os.path.exists(cpath):
        raw = open(cpath, "rb").read()
    else:
        url = url_tpl.format(z=z, x=x, y=y)
        req = urllib.request.Request(url, headers={"User-Agent": "orchard-layout/0.1"})
        raw = None
        for attempt in range(retries):
            try:
                raw = urllib.request.urlopen(req, timeout=60).read()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"tile fetch failed {url}: {e}")
                time.sleep(1.5 * (attempt + 1))
        with open(cpath, "wb") as f:
            f.write(raw)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        arr = np.full((256, 256, 3), 128, np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def fetch_aerial(lat, lon, radius_m, source="gsi", zoom=None):
    """Download + stitch tiles covering a square of +/- radius_m around lat/lon."""
    url_tpl, max_zoom = TILE_SOURCES[source]
    z = max_zoom if zoom is None else min(zoom, max_zoom)
    mpp = metres_per_pixel(lat, z)
    r_px = radius_m / mpp
    cx, cy = _tile_xy(lat, lon, z)
    x_min = int(math.floor(cx - r_px / 256.0)); x_max = int(math.floor(cx + r_px / 256.0))
    y_min = int(math.floor(cy - r_px / 256.0)); y_max = int(math.floor(cy + r_px / 256.0))
    rows = []
    for ty in range(y_min, y_max + 1):
        rows.append(np.concatenate([_fetch_tile(source, z, tx, ty) for tx in range(x_min, x_max + 1)], axis=1))
    mosaic = np.concatenate(rows, axis=0)
    return GeoImage(mosaic, z, x_min, y_min, source)


def crop_geoimage(gi, lat, lon, radius_m):
    px, py = gi.latlon_to_pix(lat, lon)
    r = radius_m / gi.mpp
    h, w = gi.img.shape[:2]
    x0, x1 = max(0, int(px - r)), min(w, int(px + r))
    y0, y1 = max(0, int(py - r)), min(h, int(py + r))
    sub = GeoImage(gi.img[y0:y1, x0:x1].copy(), gi.z, gi.x0 + x0 / 256.0, gi.y0 + y0 / 256.0, gi.source)
    return sub


# ─────────────────────────────────────────────────────────────────────────────
#  Crown detection (vegetation index + blob local maxima)
# ─────────────────────────────────────────────────────────────────────────────
def detect_crowns(gi, crown_r_m=2.0, min_spacing_m=3.0, veg_thresh=None, roi_poly_latlon=None):
    """
    Returns list of dicts {px, py, lat, lon, crown_r_m, score} for likely tree crowns.

    Method: excess-green index (2G-R-B) -> blur at crown scale -> local maxima at least
    min_spacing apart -> keep maxima above veg_thresh (Otsu on the index if None).
    It is deliberately simple; always check the preview PNG and prune with --roi or KML.
    """
    img = gi.img.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    exg = (2 * G - R - B) / (R + G + B + 1.0)          # -1..1, >0 is green-ish
    mpp = gi.mpp
    sigma = max(1.0, 0.5 * crown_r_m / mpp)
    blur = cv2.GaussianBlur(exg, (0, 0), sigma)

    # suppress strongly dark/shadow and very bright (roofs, concrete)
    lum = (R + G + B) / 3.0
    blur[lum < 25] = -1.0
    blur[lum > 215] = -1.0

    if roi_poly_latlon:
        mask = np.zeros(blur.shape, np.uint8)
        pts = np.array([gi.latlon_to_pix(la, lo) for la, lo in roi_poly_latlon], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        blur[mask == 0] = -1.0

    if veg_thresh is None:
        valid = blur[blur > -1.0]
        u8 = np.clip((valid + 1.0) * 127.5, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        veg_thresh = t / 127.5 - 1.0

    k = max(3, int(round(min_spacing_m / mpp)) | 1)
    dil = cv2.dilate(blur, np.ones((k, k), np.uint8))
    peaks = np.argwhere((blur >= dil - 1e-6) & (blur > veg_thresh))

    out = []
    for py, px in peaks:
        lat, lon = gi.pix_to_latlon(px + 0.5, py + 0.5)
        out.append({"px": float(px), "py": float(py), "lat": lat, "lon": lon,
                    "crown_r_m": float(crown_r_m), "score": float(blur[py, px])})
    out.sort(key=lambda d: -d["score"])
    return out, float(veg_thresh)


def row_structure(trees_ne):
    """Estimate dominant row heading (deg, 0 = north, 90 = east) and spacings.
    Uses nearest-neighbour vectors: in-row spacing is the modal NN distance; the row
    heading is the modal NN direction (mod 180)."""
    pts = np.array([(n, e) for n, e in trees_ne], np.float64)
    if len(pts) < 3:
        return None
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(axis=1)
    vec = pts[nn] - pts
    ang = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 180.0   # heading from north
    hist, edges = np.histogram(ang, bins=36, range=(0, 180))
    heading = float(edges[hist.argmax()] + 2.5)
    in_row = float(np.median(dist[np.arange(len(pts)), nn]))
    # row spacing: NN distance in the direction perpendicular to heading
    h = math.radians(heading)
    perp = np.array([-math.sin(h), math.cos(h)])
    along = np.array([math.cos(h), math.sin(h)])
    proj_perp = np.abs(d @ perp)
    proj_along = np.abs(d @ along)
    cand = np.where((proj_along < in_row * 0.6) & (proj_perp > in_row * 0.5), proj_perp, np.inf)
    row_sp = cand.min(axis=1)
    row_sp = row_sp[np.isfinite(row_sp)]
    row = float(np.median(row_sp)) if len(row_sp) else float("nan")
    return {"row_heading_deg": heading, "in_row_m": in_row, "row_m": row}


# ─────────────────────────────────────────────────────────────────────────────
#  KML (Google Earth) import
# ─────────────────────────────────────────────────────────────────────────────
def _parse_coords(text):
    pts = []
    for tok in text.strip().split():
        parts = tok.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            pts.append((lat, lon))
    return pts


def parse_kml(path):
    """Returns {'points': [(lat,lon,name)], 'lines': [[(lat,lon),...]], 'polygons': [[(lat,lon),...]]}"""
    tree = ET.parse(path)
    root = tree.getroot()
    out = {"points": [], "lines": [], "polygons": []}
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name = pm.findtext("k:name", default="", namespaces=KML_NS)
        for pt in pm.iter("{http://www.opengis.net/kml/2.2}Point"):
            c = pt.findtext("k:coordinates", default="", namespaces=KML_NS)
            p = _parse_coords(c)
            if p:
                out["points"].append((p[0][0], p[0][1], name))
        for ls in pm.iter("{http://www.opengis.net/kml/2.2}LineString"):
            c = ls.findtext("k:coordinates", default="", namespaces=KML_NS)
            out["lines"].append(_parse_coords(c))
        for pg in pm.iter("{http://www.opengis.net/kml/2.2}Polygon"):
            ob = pg.find(".//k:outerBoundaryIs//k:coordinates", KML_NS)
            if ob is not None:
                out["polygons"].append(_parse_coords(ob.text))
    return out


def trees_along_line(line_ne, spacing_m, end_margin_m=0.0):
    """Place trees every spacing_m along a polyline given in (north, east) metres."""
    pts = [np.array(p, np.float64) for p in line_ne]
    seg_len = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    total = sum(seg_len)
    n = int((total - 2 * end_margin_m) // spacing_m) + 1
    if n < 1:
        return []
    start = (total - (n - 1) * spacing_m) / 2.0
    out = []
    for i in range(n):
        s = start + i * spacing_m
        acc = 0.0
        for j, L in enumerate(seg_len):
            if s <= acc + L or j == len(seg_len) - 1:
                t = 0.0 if L == 0 else (s - acc) / L
                p = pts[j] + t * (pts[j + 1] - pts[j])
                out.append((float(p[0]), float(p[1])))
                break
            acc += L
    return out


def _point_in_poly(n, e, poly_ne):
    inside = False
    m = len(poly_ne)
    for i in range(m):
        n1, e1 = poly_ne[i]
        n2, e2 = poly_ne[(i + 1) % m]
        if (e1 > e) != (e2 > e):
            x = (e - e1) * (n2 - n1) / (e2 - e1) + n1
            if n < x:
                inside = not inside
    return inside


def fill_polygon_rows(poly_ne, row_spacing_m, in_row_m, heading_deg=None, edge_margin_m=1.5):
    """Fill a boundary polygon (north, east metres) with straight rows.
    heading_deg: direction of the rows (0 = north). None = along the polygon's long axis."""
    P = np.array(poly_ne, np.float64)
    c = P.mean(axis=0)
    if heading_deg is None:
        cov = np.cov((P - c).T)
        w, v = np.linalg.eigh(cov)
        a = v[:, w.argmax()]
        heading_deg = math.degrees(math.atan2(a[1], a[0]))
    h = math.radians(heading_deg)
    along = np.array([math.cos(h), math.sin(h)])
    perp = np.array([-math.sin(h), math.cos(h)])
    rel = P - c
    a_min, a_max = (rel @ along).min(), (rel @ along).max()
    p_min, p_max = (rel @ perp).min(), (rel @ perp).max()
    out = []
    n_rows = int((p_max - p_min) // row_spacing_m)
    p0 = p_min + ((p_max - p_min) - n_rows * row_spacing_m) / 2.0
    for r in range(n_rows + 1):
        pp = p0 + r * row_spacing_m
        n_in = int((a_max - a_min) // in_row_m)
        a0 = a_min + ((a_max - a_min) - n_in * in_row_m) / 2.0
        for i in range(n_in + 1):
            aa = a0 + i * in_row_m
            q = c + aa * along + pp * perp
            # edge margin: shrink test by checking a few offsets
            ok = all(_point_in_poly(q[0] + dn, q[1] + de, poly_ne)
                     for dn, de in ((0, 0), (edge_margin_m, 0), (-edge_margin_m, 0),
                                    (0, edge_margin_m), (0, -edge_margin_m)))
            if ok:
                out.append((float(q[0]), float(q[1])))
    return out, float(heading_deg)


# ─────────────────────────────────────────────────────────────────────────────
#  Layout file I/O
# ─────────────────────────────────────────────────────────────────────────────
FIELDS = ["north_m", "east_m", "z_m", "yaw_deg", "scale", "crown_r_m", "lat", "lon"]


def write_layout(path, trees, meta):
    """trees: list of dicts with at least north_m, east_m. meta: dict written as '# k=v' header."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# " + " ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for t in trees:
            row = {k: t.get(k, "") for k in FIELDS}
            for k in ("z_m", "yaw_deg"):
                row[k] = row[k] if row[k] != "" else 0.0
            row["scale"] = row["scale"] if row["scale"] != "" else 1.0
            for k in ("north_m", "east_m", "z_m", "yaw_deg", "scale", "crown_r_m"):
                if row[k] != "":
                    row[k] = f"{float(row[k]):.3f}"
            for k in ("lat", "lon"):
                if row[k] != "":
                    row[k] = f"{float(row[k]):.7f}"
            w.writerow(row)


def read_layout(path):
    """Returns (trees, meta). trees: list of dicts with float fields; meta: dict of header k=v."""
    meta, rows = {}, []
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines()
    body = []
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        elif ln.strip():
            body.append(ln)
    for r in csv.DictReader(io.StringIO("\n".join(body))):
        t = {}
        for k, v in r.items():
            if v is None or v == "":
                continue
            try:
                t[k] = float(v)
            except ValueError:
                t[k] = v
        t.setdefault("z_m", 0.0); t.setdefault("yaw_deg", 0.0); t.setdefault("scale", 1.0)
        rows.append(t)
    return rows, meta


def list_layouts(pattern="*.csv", layout_dir=LAYOUT_DIR):
    return sorted(glob.glob(os.path.join(layout_dir, pattern)))


def layout_summary(trees):
    if not trees:
        return "empty"
    n = np.array([t["north_m"] for t in trees]); e = np.array([t["east_m"] for t in trees])
    rs = row_structure(list(zip(n, e)))
    s = f"{len(trees)} trees, extent {n.ptp():.0f} m N x {e.ptp():.0f} m E"
    if rs:
        s += (f", rows heading {rs['row_heading_deg']:.0f} deg, in-row {rs['in_row_m']:.1f} m, "
              f"row {rs['row_m']:.1f} m")
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Episode windows (used by orchard_env.py)
# ─────────────────────────────────────────────────────────────────────────────
def pick_window(trees, n_trees, rng=random, anchor_idx=None, min_dist_m=0.0):
    """Pick a compact group of n_trees: an anchor tree plus its nearest neighbours.

    min_dist_m > 0 thins the group so no two chosen trees are closer than that
    (the per-tree orbit reward in orchard_env.py needs ~5 m between trunks; real
    mikan rows are often 2.5-3 m, so the sim keeps the row geometry but skips
    neighbours that would be un-orbitable). Returns fewer than n_trees if the
    layout runs out of candidates."""
    if not trees:
        return []
    if anchor_idx is None:
        anchor_idx = rng.randrange(len(trees))
    a = trees[anchor_idx]
    order = sorted(range(len(trees)),
                   key=lambda i: math.hypot(trees[i]["north_m"] - a["north_m"],
                                            trees[i]["east_m"] - a["east_m"]))
    chosen = []
    for i in order:
        t = trees[i]
        if min_dist_m > 0 and any(math.hypot(t["north_m"] - c["north_m"], t["east_m"] - c["east_m"]) < min_dist_m
                                  for c in chosen):
            continue
        chosen.append(t)
        if len(chosen) >= n_trees:
            break
    return chosen


def window_to_env_frame(window, start_x, rotate_deg=0.0, min_dist_from_origin=None):
    """
    Map a window of layout trees into the environment frame used by orchard_env.py:
    the drone takes off at (0,0) facing +x. The group is rotated by rotate_deg about
    its centroid, centred on y=0, and shifted so its nearest tree sits at x=start_x.
    Returns [(x, y, z, yaw_deg, scale), ...] sorted by x.
    """
    if not window:
        return []
    N = np.array([t["north_m"] for t in window]); E = np.array([t["east_m"] for t in window])
    cn, ce = N.mean(), E.mean()
    r = math.radians(rotate_deg)
    x = (N - cn) * math.cos(r) - (E - ce) * math.sin(r)
    y = (N - cn) * math.sin(r) + (E - ce) * math.cos(r)
    x = x - x.min() + start_x
    y = y - (y.min() + y.max()) / 2.0
    out = []
    for i, t in enumerate(window):
        out.append((float(x[i]), float(y[i]), float(t.get("z_m", 0.0)),
                    float(t.get("yaw_deg", 0.0)), float(t.get("scale", 1.0))))
    if min_dist_from_origin is not None:
        # push the whole group forward if any tree would sit on top of the take-off point
        d = min(math.hypot(px, py) for px, py, *_ in out)
        if d < min_dist_from_origin:
            out = [(px + (min_dist_from_origin - d), py, pz, yw, sc) for px, py, pz, yw, sc in out]
    out.sort(key=lambda t: t[0])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Preview rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_preview(gi, trees_latlon, path, poly_latlon=None, label=None):
    """Draw detected/placed trees on the aerial image and save a PNG."""
    img = cv2.cvtColor(gi.img.copy(), cv2.COLOR_RGB2BGR)
    scale = 1 if max(img.shape[:2]) >= 1200 else int(math.ceil(1200 / max(img.shape[:2])))
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if poly_latlon:
        pts = np.array([[v * scale for v in gi.latlon_to_pix(la, lo)] for la, lo in poly_latlon], np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 255), 2)
    for t in trees_latlon:
        px, py = gi.latlon_to_pix(t["lat"], t["lon"])
        r = max(3, int(t.get("crown_r_m", 2.0) / gi.mpp * scale))
        cv2.circle(img, (int(px * scale), int(py * scale)), r, (0, 0, 255), 2)
        cv2.circle(img, (int(px * scale), int(py * scale)), 2, (255, 255, 0), -1)
    if label:
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    bar = 20.0 / gi.mpp * scale  # 20 m scale bar
    h = img.shape[0]
    cv2.line(img, (10, h - 15), (int(10 + bar), h - 15), (255, 255, 255), 3)
    cv2.putText(img, "20 m", (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def render_layout_plan(trees, path, title=None):
    """Top-down plan of a layout CSV (no imagery), 10 px per metre."""
    if not trees:
        return None
    n = np.array([t["north_m"] for t in trees]); e = np.array([t["east_m"] for t in trees])
    pad, ppm = 10.0, 10.0
    W = int((e.ptp() + 2 * pad) * ppm); H = int((n.ptp() + 2 * pad) * ppm)
    img = np.full((H, W, 3), 245, np.uint8)
    for t in trees:
        px = int((t["east_m"] - e.min() + pad) * ppm)
        py = int((n.max() - t["north_m"] + pad) * ppm)
        r = int(t.get("crown_r_m", 2.0) * t.get("scale", 1.0) * ppm)
        cv2.circle(img, (px, py), r, (60, 140, 60), -1)
        cv2.circle(img, (px, py), 2, (30, 30, 30), -1)
    cv2.line(img, (10, H - 15), (int(10 + 20 * ppm), H - 15), (0, 0, 0), 3)
    cv2.putText(img, "20 m   (up = north)", (10, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path
