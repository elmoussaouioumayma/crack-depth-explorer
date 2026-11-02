"""Find cracks in a height map and measure their depth, width and length in millimetres.

Crack depth is measured against the surface around the crack, not against a global
zero: a morphological closing of the height map "bridges" every valley narrower
than the structuring element, so  closing(height) - height  is the depth below the
local surface at every pixel (the black-hat transform, applied to heights instead
of grey values). Round air voids are deep too, so a shape filter keeps only
elongated regions. The crack network is then skeletonised and split into segments
between junctions, and each segment gets its own length, depth and width.

Width is measured at half the crack depth (full width at half depth). The top edge of
a real crack is often chipped and rounded, so "the" opening is not well defined, while
the width at half depth is repeatable.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, maximum_filter
from skimage import measure
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize


@dataclass
class CrackParams:
    max_width_mm: float = 7.0     # widest crack opening that is still measured as a crack
    seed_depth_mm: float = 0.4    # a crack must be at least this deep somewhere ...
    edge_depth_mm: float = 0.15   # ... and continues as long as it is this deep
    min_elongation: float = 3.0   # length / width of a region; round voids are removed
    min_segment_mm: float = 2.0   # shorter skeleton pieces are not reported


def local_depth(height, px_mm, max_width_mm):
    r = max(1, int(round(max_width_mm / px_mm / 2)))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed = cv2.morphologyEx(height.astype(np.float32), cv2.MORPH_CLOSE, se)
    return np.clip(closed - height, 0, None)


def crack_mask(depth, px_mm, p: CrackParams):
    mask = apply_hysteresis_threshold(depth, p.edge_depth_mm, p.seed_depth_mm)
    lab = measure.label(mask)
    keep = np.zeros_like(mask)
    for r in measure.regionprops(lab):
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-6)
        if elong >= p.min_elongation or (r.solidity < 0.4 and r.area * px_mm ** 2 > 20):
            keep[lab == r.label] = True
    return keep


def _segments(skel):
    """Split a skeleton at junctions (pixels with more than two neighbours)."""
    nb = convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junctions = skel & (nb > 2)
    junctions = cv2.dilate(junctions.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return measure.label(skel & ~junctions, connectivity=2), junctions & skel


def _length_px(coords):
    """Length of a thin 8-connected chain: straight links count 1, diagonal links sqrt(2).

    A diagonal link is ignored when the two pixels are already joined through a
    common straight neighbour (an L-shaped corner), so no step is counted twice."""
    s = set(map(tuple, coords))
    length = 0.0
    for y, x in s:
        length += ((y, x + 1) in s) + ((y + 1, x) in s)
        for dx in (1, -1):
            if (y + 1, x + dx) in s and (y, x + dx) not in s and (y + 1, x) not in s:
                length += np.sqrt(2)
    return max(length, 1.0)


@dataclass
class CrackResult:
    height: np.ndarray        # mm, relative to the median surface
    depth: np.ndarray         # mm below the local surface
    mask: np.ndarray          # crack pixels
    skeleton: np.ndarray      # crack centre lines
    depth_line: np.ndarray    # deepest point across the crack, at each centre-line pixel (mm)
    width_line: np.ndarray    # crack width at half depth, at each centre-line pixel (mm)
    segments: list            # one dict per crack segment
    segment_labels: np.ndarray
    px_mm: float

    def summary(self):
        if not self.segments:
            return {"crack_segments": 0}
        on = self.skeleton
        return {"crack_segments": len(self.segments),
                "total_length_mm": round(sum(s["length_mm"] for s in self.segments), 1),
                "max_depth_mm": round(float(self.depth_line[on].max()), 2),
                "median_depth_mm": round(float(np.median(self.depth_line[on])), 2),
                "max_width_mm": round(float(self.width_line[on].max()), 2),
                "median_width_mm": round(float(np.median(self.width_line[on])), 2),
                "crack_area_%": round(float(100 * self.mask.mean()), 2)}


def analyse(height, px_mm, params=CrackParams()):
    depth = local_depth(height, px_mm, params.max_width_mm)
    mask = crack_mask(depth, px_mm, params)
    skel = skeletonize(mask)
    r = max(3, int(params.max_width_mm / px_mm / 3))
    local_max = maximum_filter(depth, size=r)
    depth_line = np.where(skel, local_max, 0.0)
    # width = full width at half depth: the edge of a chipped crack is not sharp, half depth is
    half = mask & (depth >= 0.5 * local_max)
    width_line = np.where(skel, 2 * distance_transform_edt(half) * px_mm, 0.0)
    labels, _ = _segments(skel)
    segments = []
    for reg in measure.regionprops(labels):
        L = _length_px(reg.coords) * px_mm
        if L < params.min_segment_mm:
            labels[labels == reg.label] = 0
            continue
        ys, xs = reg.coords[:, 0], reg.coords[:, 1]
        d, w = depth_line[ys, xs], width_line[ys, xs]
        segments.append({"id": len(segments) + 1, "label": reg.label, "length_mm": round(L, 1),
                         "mean_depth_mm": round(float(d.mean()), 2), "max_depth_mm": round(float(d.max()), 2),
                         "mean_width_mm": round(float(w.mean()), 2), "max_width_mm": round(float(w.max()), 2),
                         "centre_xy": (int(np.median(xs)), int(np.median(ys)))})
    return CrackResult(height, depth, mask, skel, depth_line, width_line, segments, labels, px_mm)
