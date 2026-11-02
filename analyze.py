"""Measure cracks in 3D from four photos lit from four sides.

    python analyze.py demo/scene_N.png demo/scene_E.png demo/scene_S.png demo/scene_W.png --px-mm 0.1
    python analyze.py photos/*.jpg --azimuth 90 0 270 180 --elevation 40 --px-mm 0.08 --out results/wall

The photos must be given in the same order as --azimuth (direction the light comes FROM,
in degrees: 0 = right, 90 = top of the image, 180 = left, 270 = bottom).
Writes report.png, height_mm.npy, crack_segments.csv and summary.json.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from cracks import CrackParams, analyse
from photostereo import Lights, estimate_normals, integrate, load_images, remove_trend


def run(paths, azimuth=(90, 0, 270, 180), elevation=45.0, px_mm=0.1, scale=1.0, srgb=True,
        detrend_mm=None, params=CrackParams()):
    lights = Lights(tuple(float(a) for a in azimuth), float(elevation), tuple(Path(p).stem for p in paths))
    imgs = load_images(paths, srgb=srgb, scale=scale)
    px = px_mm / scale
    normals, albedo = estimate_normals(imgs, lights.directions())
    height = remove_trend(integrate(normals, px), px, detrend_mm)
    return analyse(height, px, params), imgs, albedo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--azimuth", nargs="+", type=float, default=[90, 0, 270, 180])
    ap.add_argument("--elevation", type=float, default=45.0, help="light angle above the surface (degrees)")
    ap.add_argument("--px-mm", type=float, required=True, help="pixel size in mm (measure a ruler in the photo)")
    ap.add_argument("--scale", type=float, default=1.0, help="resize the photos first (e.g. 0.25 for phone photos)")
    ap.add_argument("--linear", action="store_true", help="images are linear (raw), not sRGB/JPEG")
    ap.add_argument("--detrend-mm", type=float, default=None,
                    help="remove surface shape broader than this (helps with uneven torch light)")
    ap.add_argument("--max-width-mm", type=float, default=CrackParams.max_width_mm)
    ap.add_argument("--out", default="results/demo")
    a = ap.parse_args()
    if len(a.images) != len(a.azimuth):
        ap.error(f"{len(a.images)} images but {len(a.azimuth)} azimuth values")

    res, imgs, albedo = run(a.images, a.azimuth, a.elevation, a.px_mm, a.scale, not a.linear, a.detrend_mm,
                            CrackParams(max_width_mm=a.max_width_mm))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "height_mm.npy", res.height.astype(np.float32))
    with open(out / "crack_segments.csv", "w", newline="") as f:
        cols = ["id", "length_mm", "mean_depth_mm", "max_depth_mm", "mean_width_mm", "max_width_mm"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res.segments)
    (out / "summary.json").write_text(json.dumps(res.summary(), indent=2))
    matplotlib.use("Agg")
    from report import plot_report
    plot_report(res, imgs.mean(0), albedo, out / "report.png")
    print(json.dumps(res.summary(), indent=2))
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
