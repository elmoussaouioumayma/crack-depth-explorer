"""Check the measurements against scenes whose true 3D shape is known.

    python validate.py            # about 10 minutes; writes results/validation.json and images/validation*.png

1. Accuracy on 20 unseen scenes: crack depth and width at every true crack centre-line
   pixel, crack detection, and the 'darker = deeper' single-image shortcut for comparison.
   The shortcut is given its best possible scale factor (fitted on the true depths), so it
   is compared at its very best; photometric stereo gets no help from the truth.
2. Robustness on 8 scenes: wrong light angle, more sensor noise, only three lights.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

from cracks import analyse
from photostereo import Lights, estimate_normals, integrate, remove_trend, single_image_shading, to_linear
from render import half_depth_width, make_scene, render

PX = 0.1
LIGHTS = Lights()


def reconstruct(imgs, light_dirs):
    stack = np.stack([to_linear(i) for i in imgs])
    n, _ = estimate_normals(stack, light_dirs)
    return analyse(remove_trend(integrate(n, PX), PX), PX)


def sample(res, truth_depth, truth_width, border=10):
    """Measured vs true depth and width at true crack centre-line pixels (depth >= 0.3 mm)."""
    sel = truth_depth >= 0.3
    sel[:border] = sel[-border:] = False
    sel[:, :border] = sel[:, -border:] = False
    est_depth = maximum_filter(res.depth, size=7)[sel]
    # width: from the nearest measured centre-line pixel (if one is within 0.5 mm)
    dist, (iy, ix) = distance_transform_edt(~res.skeleton, return_indices=True)
    found = dist[sel] <= 5
    est_width = res.width_line[iy[sel], ix[sel]]
    return dict(true_depth=truth_depth[sel], true_width=half_depth_width(truth_width[sel]),
                opening=truth_width[sel], est_depth=est_depth, est_width=est_width, found=found)


def detection(res, truth_depth, tol_px=5):
    """Precision: measured centre lines that lie on a real crack (any depth).
    Recall: centre lines of cracks deeper than 0.4 mm (the detection limit) that were found."""
    near_crack = distance_transform_edt(~(truth_depth > 0)) <= tol_px
    deep_line = truth_depth >= 0.4
    near_est = distance_transform_edt(~res.skeleton) <= tol_px
    precision = (res.skeleton & near_crack).sum() / max(res.skeleton.sum(), 1)
    recall = (deep_line & near_est).sum() / max(deep_line.sum(), 1)
    return float(precision), float(recall)


def cat(rows, key):
    return np.concatenate([r[key] for r in rows])


def main(n_scenes=20, n_robust=8):
    out = {}
    rows, base_rows, det = [], [], []
    print("1) accuracy")
    for seed in range(100, 100 + n_scenes):
        height, albedo, cdepth, cwidth, _ = make_scene(seed, px_mm=PX)
        imgs = render(height, albedo, LIGHTS.directions(), PX, seed=seed)
        res = reconstruct(imgs, LIGHTS.directions())
        rows.append(sample(res, cdepth, cwidth))
        det.append(detection(res, cdepth))
        # single photo with the light at the camera (ring light / flash next to the lens)
        top = to_linear(render(height, albedo, [0, 0, 1], PX, seed=seed)[0])
        sel = (cdepth >= 0.3)
        base_rows.append(dict(idx=maximum_filter(single_image_shading(top), size=7)[sel], true=cdepth[sel]))
        print(f"  scene {seed}: precision {det[-1][0]:.2f} recall {det[-1][1]:.2f}")

    td, ed = cat(rows, "true_depth"), cat(rows, "est_depth")
    tw, ew, found = cat(rows, "true_width"), cat(rows, "est_width"), cat(rows, "found")
    ratio = td / cat(rows, "opening")
    bi, bt = np.concatenate([b["idx"] for b in base_rows]), np.concatenate([b["true"] for b in base_rows])
    scale = float((bi @ bt) / (bi @ bi))                            # best possible calibration
    base_err = np.abs(scale * bi - bt)
    err = np.abs(ed - td)
    out["accuracy"] = {
        "scenes": n_scenes, "centre_line_points": int(td.size),
        "true_depth_range_mm": [round(float(td.min()), 2), round(float(td.max()), 2)],
        "photometric_stereo": {
            "depth_mae_mm": round(float(err.mean()), 3), "depth_median_abs_error_mm": round(float(np.median(err)), 3),
            "depth_bias_mm": round(float((ed - td).mean()), 3),
            "depth_relative_error_median_%": round(float(100 * np.median(err / td)), 1),
            "depth_correlation": round(float(np.corrcoef(td, ed)[0, 1]), 3),
            "width_mae_mm": round(float(np.abs(ew - tw)[found].mean()), 3),
            "width_median_abs_error_mm": round(float(np.median(np.abs(ew - tw)[found])), 3),
            "detection_precision": round(float(np.mean([d[0] for d in det])), 3),
            "detection_recall": round(float(np.mean([d[1] for d in det])), 3)},
        "single_image_shading_best_case": {
            "depth_mae_mm": round(float(base_err.mean()), 3),
            "depth_median_abs_error_mm": round(float(np.median(base_err)), 3),
            "depth_correlation": round(float(np.corrcoef(bt, bi)[0, 1]), 3),
            "fitted_scale_mm_per_index": round(scale, 2)}}
    bins = [0, 0.4, 0.6, 0.8, 1.2]
    out["depth_error_by_depth_to_width_ratio"] = {
        f"{lo}-{hi}": {"points": int(((ratio >= lo) & (ratio < hi)).sum()),
                       "median_abs_error_mm": round(float(np.median(err[(ratio >= lo) & (ratio < hi)])), 3),
                       "bias_mm": round(float((ed - td)[(ratio >= lo) & (ratio < hi)].mean()), 3)}
        for lo, hi in zip(bins[:-1], bins[1:]) if ((ratio >= lo) & (ratio < hi)).sum() > 50}

    print("2) robustness")
    rob = {"assumed_elevation_deg": {}, "noise": {}, "three_lights": {}}
    scenes = [make_scene(s, px_mm=PX) for s in range(200, 200 + n_robust)]

    def run(true_el=45.0, assumed_el=45.0, noise=0.01, use=(0, 1, 2, 3)):
        e = []
        for i, (height, albedo, cdepth, cwidth, _) in enumerate(scenes):
            imgs = render(height, albedo, LIGHTS.directions(true_el), PX, noise=noise, seed=i)
            res = reconstruct([imgs[j] for j in use], LIGHTS.directions(assumed_el)[list(use)])
            r = sample(res, cdepth, cwidth)
            e.append(r["est_depth"] - r["true_depth"])
        e = np.concatenate(e)
        return {"depth_mae_mm": round(float(np.abs(e).mean()), 3), "depth_bias_mm": round(float(e.mean()), 3)}

    for true_el in (35, 40, 45, 50, 55):
        rob["assumed_elevation_deg"][f"true {true_el}, assumed 45"] = run(true_el=true_el)
        print("  elevation", true_el, rob["assumed_elevation_deg"][f"true {true_el}, assumed 45"])
    for noise in (0.01, 0.03, 0.06):
        rob["noise"][f"sigma {noise}"] = run(noise=noise)
        print("  noise", noise, rob["noise"][f"sigma {noise}"])
    rob["three_lights"]["N, E, S"] = run(use=(0, 1, 2))
    print("  three lights", rob["three_lights"])
    out["robustness"] = rob

    Path("results").mkdir(exist_ok=True)
    Path("results/validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["accuracy"], indent=2))

    # figure
    Path("images").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    k = rng.choice(td.size, min(4000, td.size), replace=False)
    kb = rng.choice(bt.size, min(4000, bt.size), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    lim = float(td.max()) * 1.05
    ax[0].scatter(bt[kb], scale * bi[kb], s=3, alpha=0.25, color="#9aa5b1")
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="true depth (mm)", ylabel="estimated depth (mm)",
              title=f"One photo, 'darker = deeper'\n(best-case scale)  MAE {base_err.mean():.2f} mm")
    ax[1].scatter(td[k], ed[k], s=3, alpha=0.25, color="#d6452f")
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set(xlim=(0, lim), ylim=(0, lim), xlabel="true depth (mm)", ylabel="estimated depth (mm)",
              title=f"Photometric stereo, 4 photos\n(no calibration)  MAE {err.mean():.2f} mm")
    labels = list(out["depth_error_by_depth_to_width_ratio"])
    vals = [out["depth_error_by_depth_to_width_ratio"][l]["bias_mm"] for l in labels]
    ax[2].bar(labels, vals, color="#d6452f")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="crack depth / opening width", ylabel="mean depth error (mm)",
              title="Narrow, deep cracks: the bottom is in shadow\nin most photos, so depth is underestimated")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("images/validation.png", dpi=120)

    plot_robustness(rob)


def plot_robustness(rob):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    el = rob["assumed_elevation_deg"]
    ax[0].plot([35, 40, 45, 50, 55], [v["depth_bias_mm"] for v in el.values()], "o-", color="#d6452f")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="true light elevation (°), always assumed 45°", ylabel="mean depth error (mm)",
              title="Wrong light angle")
    nz = rob["noise"]
    ax[1].plot([0.01, 0.03, 0.06], [v["depth_mae_mm"] for v in nz.values()], "o-", color="#d6452f")
    ax[1].set(xlabel="sensor noise (fraction of white)", ylabel="depth MAE (mm)", title="Image noise",
              ylim=(0, 1.3 * max(v["depth_mae_mm"] for v in nz.values())))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("images/robustness.png", dpi=120)


if __name__ == "__main__":
    main()
