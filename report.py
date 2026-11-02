"""One-page figure of a crack analysis: photo, colour, 3D height, crack depth map,
cross-sections through the deepest cracks and the table of crack segments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LightSource

DEPTH_CMAP = "magma_r"


def cross_section(height, centre, direction, half_len_px, px_mm):
    """Height profile perpendicular to a crack through `centre` (x, y)."""
    from scipy.ndimage import map_coordinates
    dx, dy = -direction[1], direction[0]
    t = np.arange(-half_len_px, half_len_px + 1)
    xs, ys = centre[0] + t * dx, centre[1] + t * dy
    z = map_coordinates(height, [ys, xs], order=1, mode="nearest")
    return t * px_mm, z, xs, ys


def _direction(skeleton_labels, label, centre, r=8):
    ys, xs = np.nonzero(skeleton_labels == label)
    near = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= r * r
    pts = np.c_[xs[near], ys[near]].astype(float)
    if len(pts) < 3:
        return np.array([1.0, 0.0])
    pts -= pts.mean(0)
    v = np.linalg.svd(pts, full_matrices=False)[2][0]
    return v / np.linalg.norm(v)


def deepest_points(res, n=3, half_mm=4.0):
    """For the deepest crack segments: the deepest centre-line point whose cross-section
    stays inside the image and crosses only this one crack (no junction, no second crack)."""
    from scipy.ndimage import map_coordinates
    H, W = res.height.shape
    half = int(half_mm / res.px_mm)
    out = []
    for s in sorted(res.segments, key=lambda s: -s["max_depth_mm"]):
        ys, xs = np.nonzero(res.segment_labels == s["label"])
        for k in np.argsort(-res.depth_line[ys, xs])[:60]:
            c = (int(xs[k]), int(ys[k]))
            d = _direction(res.segment_labels, s["label"], c)
            _, _, lx, ly = cross_section(res.height, c, d, half, res.px_mm)
            if lx.min() < 0 or ly.min() < 0 or lx.max() > W - 1 or ly.max() > H - 1:
                continue
            on = map_coordinates(res.mask.astype(float), [ly, lx], order=0) > 0.5
            if np.count_nonzero(np.diff(on.astype(int)) == 1) + on[0] == 1 and not on[0] and not on[-1]:
                out.append((s, c, d))
                break
        if len(out) == n:
            break
    return out


def plot_report(res, photo, albedo, out=None, title="Crack analysis from 4 photos (photometric stereo)"):
    px = res.px_mm
    H, W = res.height.shape
    ext = [0, W * px, H * px, 0]
    fig = plt.figure(figsize=(17, 10.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.32,
                           left=0.04, right=0.98, top=0.92, bottom=0.06)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(photo, cmap="gray", extent=ext)
    ax.set_title("Average of the 4 photos")
    ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(albedo, cmap="gray", extent=ext)
    ax.set_title("Surface colour (albedo)\nshading removed")
    ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[0, 2:], projection="3d")
    step = max(1, int(max(H, W) / 260))
    Z = res.height[::step, ::step]
    X, Y = np.meshgrid(np.arange(Z.shape[1]) * px * step, np.arange(Z.shape[0]) * px * step)
    rgb = LightSource(315, 45).shade(Z, plt.colormaps["cividis"], vert_exag=4, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=42, azim=-60)
    ax.set_box_aspect([W, H, 0.12 * max(H, W)], zoom=1.35)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title("3D surface (height x4 for visibility)")

    ax = fig.add_subplot(gs[1, 0])
    show = np.where(res.mask, res.depth, np.nan)
    ax.imshow(photo, cmap="gray", extent=ext, alpha=0.6)
    im = ax.imshow(show, cmap=DEPTH_CMAP, extent=ext, vmin=0, vmax=max(np.nanmax(show), 0.5) if res.mask.any() else 1)
    pts = deepest_points(res)
    colours = ["#1f77b4", "#2ca02c", "#d62728"]
    for (s, c, d), col in zip(pts, colours):
        _, _, xs, ys = cross_section(res.height, c, d, int(4 / px), px)
        ax.plot(xs * px, ys * px, color=col, lw=2)
        ax.text(c[0] * px, c[1] * px, f" {s['id']}", color=col, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.18, label="depth below surface (mm)")
    ax.set_title("Crack depth (lines = cross-sections)")
    ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[1, 1])
    lo = 0.0
    for (s, c, d), col in zip(pts, colours):
        t, z, _, _ = cross_section(res.height, c, d, int(4 / px), px)
        ax.plot(t, z, color=col, lw=2, label=f"segment {s['id']}")
        lo = min(lo, z.min())
    ax.set_ylim(lo * 1.45, 0.4)
    ax.set_xlabel("distance across the crack (mm)")
    ax.set_ylabel("height (mm)")
    ax.set_title("Cross-sections through the deepest cracks")
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=3)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    segs = sorted(res.segments, key=lambda s: -s["length_mm"])[:10]
    rows = [[s["id"], f"{s['length_mm']:.1f}", f"{s['mean_depth_mm']:.2f}", f"{s['max_depth_mm']:.2f}",
             f"{s['mean_width_mm']:.2f}", f"{s['max_width_mm']:.2f}"] for s in segs]
    if rows:
        tab = ax.table(cellText=rows, colLabels=["segment", "length", "mean depth", "max depth", "mean width",
                                                 "max width"], loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.35)
    summ = res.summary()
    ax.set_title(f"Longest crack segments (mm; width at half depth)   |   "
                 f"{summ.get('crack_segments', 0)} segments, total length {summ.get('total_length_mm', 0)} mm",
                 fontsize=10)
    if out:
        fig.savefig(out, dpi=110)
    return fig
