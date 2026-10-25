"""
Tree detectors. Each returns a list of Detection with a bearing (deg, + = right of
centre) and a distance estimate (m), sorted nearest first.

GreenBlobDetector  - RGB only. Segments canopy green in HSV, takes connected blobs,
                     distance from the blob's pixel width and the known crown diameter.
                     Works on the sim trees and on real trees against sky / soil. It is
                     the one that runs on the Tello. Swap in a learned detector by
                     subclassing and overriding detect() if the orchard background is
                     too green for it.
DepthBlobDetector  - sim only (or any depth camera). Removes the ground plane using the
                     camera height, blobs what is left, distance = median depth.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import OrbitConfig


@dataclass
class Detection:
    bearing_deg: float     # + = tree is to the right of the image centre
    dist_m: float
    elev_deg: float        # + = above image centre
    bbox: tuple            # x, y, w, h in pixels
    partial: bool          # touches the left/right image edge -> distance unreliable
    area_px: int
    score: float = 1.0


def focal_px(frame_w, hfov_deg):
    return (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _split_wide(mask, x, y, w, h):
    """Two canopies side by side merge into one wide blob. A canopy is about as wide as
    it is tall, so a blob much wider than tall is cut at the deepest valleys of its
    column profile. Returns a list of (x, y, w, h) pieces."""
    if w < 1.45 * h:
        return [(x, y, w, h)]
    col = (mask[y:y + h, x:x + w] > 0).sum(axis=0).astype(np.float32)
    col = cv2.blur(col.reshape(1, -1), (1, max(3, h // 6))).ravel()
    peak = col.max()
    pieces, start = [], 0
    i = int(0.3 * h)
    while i < w - int(0.3 * h):
        lo, hi = i - h // 4, i + h // 4
        if col[i] < 0.55 * peak and col[i] <= col[max(0, lo):hi + 1].min() + 1e-6 and (i - start) > 0.5 * h:
            pieces.append((x + start, y, i - start, h))
            start = i
            i += h // 2
        i += 1
    pieces.append((x + start, y, w - start, h))
    return pieces


def _blobs(mask, min_area):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        for px, py, pw, ph in _split_wide(mask, int(x), int(y), int(w), int(h)):
            sub = mask[py:py + ph, px:px + pw] > 0
            a = int(sub.sum())
            if a < min_area:
                continue
            ys, xs = np.nonzero(sub)
            # tighten the box to the piece's own pixels
            bx, by = px + int(xs.min()), py + int(ys.min())
            bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            out.append((bx, by, bw, bh, a, px + float(xs.mean()), py + float(ys.mean())))
    return out


class GreenBlobDetector:
    def __init__(self, cfg: OrbitConfig):
        self.cfg = cfg
        self.w, self.h = cfg.frame_w, cfg.frame_h
        self.f = focal_px(cfg.frame_w, cfg.hfov_deg)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(self.cfg.hsv_lo, np.uint8), np.array(self.cfg.hsv_hi, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return m

    def detect(self, bgr, depth=None, cam_height_m=None, pitch_deg=0.0):
        """cam_height_m: camera height above the ground. With it, range comes from the row
        where the canopy's lower edge sits (ground-plane geometry), which does not care
        whether neighbouring canopies overlap in the image. Without it, or when the blob
        runs off the bottom of the frame, range falls back to the blob width.
        pitch_deg: body pitch, nose-up positive; the camera is body-fixed so the horizon
        moves in the image when the drone leans into its motion."""
        h, w = bgr.shape[:2]
        if (w, h) != (self.w, self.h):
            self.w, self.h = w, h
            self.f = focal_px(w, self.cfg.hfov_deg)
        m = self.mask(bgr)
        dets = []
        for x, y, bw, bh, area, cx, cy in _blobs(m, self.cfg.min_blob_frac * w * h):
            partial = x <= 1 or x + bw >= w - 1
            bearing = math.degrees(math.atan2(cx - w / 2.0, self.f))
            elev = -math.degrees(math.atan2(cy - h / 2.0, self.f))
            # width-based range; a partial blob is at least this close
            width_px = max(bw, 1)
            dist = self.cfg.crown_diameter_m * self.f / width_px / math.cos(math.radians(bearing))
            drop = cam_height_m - self.cfg.canopy_bottom_m if cam_height_m is not None else 0.0
            if drop > 0.2 and y + bh < h - 2:
                # bottom of THIS tree: lowest mask pixel in the columns around its centre
                c0, c1 = int(max(x, cx - 0.15 * bw)), int(min(x + bw, cx + 0.15 * bw) + 1)
                col = m[y:y + bh, c0:c1]
                rows = np.nonzero(col.any(axis=1))[0]
                bottom = y + (rows.max() if rows.size else bh - 1) + 0.5
                ang = math.atan2(bottom - h / 2.0, self.f) - math.radians(pitch_deg)   # below the true horizon
                if ang > math.radians(2.0):
                    dist = drop / math.tan(ang) / math.cos(math.radians(bearing))
                    partial = False                                 # range is trustworthy now
            if depth is not None:
                # if a depth image is available, trust its median inside the blob instead
                sub = depth[y:y + bh, x:x + bw]
                valid = sub[(sub > 0.3) & (sub < self.cfg.max_range_m)]
                if valid.size > 20:
                    dist = float(np.median(valid))
            if dist > self.cfg.max_range_m * 1.5:
                continue
            dets.append(Detection(bearing, float(dist), elev, (x, y, bw, bh), partial, area,
                                  score=float(area) / (w * h)))
        dets.sort(key=lambda d: d.dist_m)
        return dets


class DepthBlobDetector:
    """Ground-plane removal + blobs on a planar depth image (metres)."""

    def __init__(self, cfg: OrbitConfig, cam_height_m=None):
        self.cfg = cfg
        self.w, self.h = cfg.frame_w, cfg.frame_h
        self.f = focal_px(cfg.frame_w, cfg.hfov_deg)
        self.cam_height_m = cam_height_m if cam_height_m is not None else cfg.cruise_alt_m
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask(self, depth):
        h, w = depth.shape
        rows = np.arange(h, dtype=np.float32) - h / 2.0
        ang_below = np.arctan2(rows, self.f)                     # + below horizon
        with np.errstate(divide="ignore"):
            ground = np.where(ang_below > 0.01, self.cam_height_m / np.tan(np.maximum(ang_below, 1e-3)), np.inf)
        ground = np.repeat(ground[:, None], w, axis=1)
        m = (depth < self.cfg.max_range_m) & (depth < 0.8 * ground) & (depth > 0.3)
        m = m.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        return m

    def detect(self, bgr, depth, cam_height_m=None, pitch_deg=0.0):
        if depth is None:
            return []
        if cam_height_m is not None:
            self.cam_height_m = cam_height_m
        h, w = depth.shape
        if (w, h) != (self.w, self.h):
            self.w, self.h = w, h
            self.f = focal_px(w, self.cfg.hfov_deg)
        m = self.mask(depth)
        dets = []
        for x, y, bw, bh, area, cx, cy in _blobs(m, self.cfg.min_blob_frac * w * h):
            c0, c1 = int(max(x, cx - 0.15 * bw)), int(min(x + bw, cx + 0.15 * bw) + 1)
            sub = depth[y:y + bh, c0:c1]
            msub = m[y:y + bh, c0:c1] > 0
            valid = sub[msub & (sub > 0.3) & (sub < self.cfg.max_range_m)]
            if valid.size < 10:
                continue
            # the trunk is ~half a crown behind the nearest canopy surface
            dist = float(np.percentile(valid, 25)) + 0.5 * self.cfg.crown_diameter_m * 0.5
            bearing = math.degrees(math.atan2(cx - w / 2.0, self.f))
            elev = -math.degrees(math.atan2(cy - h / 2.0, self.f))
            partial = x <= 1 or x + bw >= w - 1
            dets.append(Detection(bearing, dist, elev, (x, y, bw, bh), partial, area, score=float(area) / (w * h)))
        dets.sort(key=lambda d: d.dist_m)
        return dets


def draw(bgr, dets, target=None, text_lines=()):
    """Annotate a frame in place: all detections grey, the tracked target green."""
    for d in dets:
        x, y, w, h = d.bbox
        col = (0, 220, 0) if d is target else (180, 180, 180)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), col, 2)
        cv2.putText(bgr, f"{d.dist_m:.1f}m {d.bearing_deg:+.0f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for i, t in enumerate(text_lines):
        cv2.putText(bgr, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    h, w = bgr.shape[:2]
    cv2.line(bgr, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    return bgr
