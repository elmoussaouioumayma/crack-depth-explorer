"""Interactive cross-section tool for the measured 3D surface.

Click two points on the image to draw a line; the height profile along that line (in
mm) appears below the image (up to 8 profiles, each in its own colour).

    Image       click point 1, then point 2  -> new profile
    Profile     hover                        -> crosshair at that position on the image
                click two points             -> measure the vertical (depth) or
                                                horizontal (width) distance between them
    Keys        u   undo the last profile
                c   clear all profiles
                s   save the figure (PNG) and all profiles (CSV)

Run as a script (same arguments as analyze.py):
    python profile_tool.py demo/scene_N.png demo/scene_E.png demo/scene_S.png demo/scene_W.png --px-mm 0.1

or in Jupyter (needs `pip install ipympl`):
    %matplotlib widget
    from profile_tool import ProfileTool
    ProfileTool(result, photo).show()
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

from analyze import run

COLOURS = ["#FF3333", "#33AAFF", "#33DD77", "#FF9933", "#CC44FF", "#FF44CC", "#E6D200", "#00CCBB"]
BG, PANEL = "#06060e", "#0d0d1a"
MAX_PROFILES, PER_ROW = 8, 4


def extract_profile(height, p1, p2, px_mm, smooth=1.0):
    """Height along the line p1 -> p2 (points as (x, y) pixels); distance in mm."""
    (x0, y0), (x1, y1) = p1, p2
    n = max(2, int(np.hypot(x1 - x0, y1 - y0)))
    xs = np.clip(np.linspace(x0, x1, n).round().astype(int), 0, height.shape[1] - 1)
    ys = np.clip(np.linspace(y0, y1, n).round().astype(int), 0, height.shape[0] - 1)
    return np.arange(n) * px_mm, gaussian_filter(height[ys, xs].astype(float), smooth), xs, ys


class ProfileTool:
    def __init__(self, result, photo, title="3D crack profiles"):
        self.res, self.photo = result, photo
        self.xunit = self.zunit = "mm"
        z = result.height
        pad = 0.08 * (z.max() - z.min() + 1e-9)
        self.zlim = (z.min() - pad, z.max() + pad)

        self.fig = plt.figure(figsize=(16, 10), facecolor=BG)
        self.fig.suptitle(f"{title}  |  click 2 points on the image  |  hover a profile -> crosshair  |  "
                          "click 2 points on a profile to measure  |  u = undo, c = clear, s = save",
                          color="#aaaaaa", fontsize=9, y=0.995)
        rows = (MAX_PROFILES + PER_ROW - 1) // PER_ROW
        gs = gridspec.GridSpec(1 + rows, PER_ROW, figure=self.fig, height_ratios=[3.4] + [2] * rows,
                               hspace=0.5, wspace=0.28, left=0.05, right=0.98, top=0.96, bottom=0.05)
        self.ax_img = self.fig.add_subplot(gs[0, :], facecolor=PANEL)
        self.ax_img.imshow(photo, cmap="gray")
        self.ax_img.tick_params(colors="white", labelsize=7)
        self.prof_axes = []
        for i in range(MAX_PROFILES):
            ax = self.fig.add_subplot(gs[1 + i // PER_ROW, i % PER_ROW], facecolor=PANEL)
            ax.set_visible(False)
            self.prof_axes.append(ax)

        self.pending = None          # first click of a new line
        self.preview = None
        self.profiles = []           # dicts: p1, p2, dist, prof, xs, ys, colour, artists
        self.meas_first = {}         # profile index -> first measurement click
        self.cross = []

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_click)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("key_press_event", self.on_key)

    # ------------------------------------------------------------------ drawing
    def _draw_profile(self, i):
        p, ax = self.profiles[i], self.prof_axes[i]
        d, z, col = p["dist"], p["prof"], p["colour"]
        ax.cla(); ax.set_visible(True); ax.set_facecolor(PANEL)
        ax.fill_between(d, 0, z, where=z >= 0, color=col, alpha=0.18)
        ax.fill_between(d, 0, z, where=z < 0, color=col, alpha=0.35)
        ax.plot(d, z, color=col, lw=1.6)
        ax.axhline(0, color="#888", lw=0.9, ls="--")
        lo, hi = int(np.argmin(z)), int(np.argmax(z))
        ax.plot(d[lo], z[lo], "v", color="#ff5555", ms=7)
        ax.plot(d[hi], z[hi], "^", color="#55ff88", ms=7)
        ax.text(0.02, 0.04, f"min {z[lo]:.2f}  max {z[hi]:+.2f} {self.zunit}", transform=ax.transAxes,
                color="white", fontsize=7, family="monospace")
        ax.set_xlim(0, d[-1]); ax.set_ylim(*self.zlim)
        ax.set_title(f"Profile {i + 1}  |  length {d[-1]:.1f} {self.xunit}", color=col, fontsize=9)
        ax.set_xlabel(f"Distance ({self.xunit})", color="white", fontsize=8)
        ax.set_ylabel(f"Height ({self.zunit})", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(color="#222233", lw=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(col)

    def add_profile(self, p1, p2):
        if len(self.profiles) >= MAX_PROFILES:
            print(f"Maximum of {MAX_PROFILES} profiles reached (press u or c).")
            return
        i = len(self.profiles)
        col = COLOURS[i % len(COLOURS)]
        dist, prof, xs, ys = extract_profile(self.res.height, p1, p2, self.res.px_mm)
        line, = self.ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=1.4)
        ends, = self.ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color=col, ms=4)
        label = self.ax_img.text((p1[0] + p2[0]) / 2 + 5, (p1[1] + p2[1]) / 2 - 6, str(i + 1),
                                 color=col, fontsize=9, fontweight="bold")
        self.profiles.append(dict(p1=p1, p2=p2, dist=dist, prof=prof, xs=xs, ys=ys, colour=col,
                                  artists=[line, ends, label]))
        self._draw_profile(i)
        self.fig.canvas.draw_idle()

    def _clear_cross(self):
        for a in self.cross:
            a.remove()
        self.cross = []

    def _clear_preview(self):
        if self.preview is not None:
            self.preview.remove()
            self.preview = None

    # ------------------------------------------------------------------ events
    def on_click(self, ev):
        if ev.inaxes is self.ax_img and ev.xdata is not None:
            pt = (int(round(ev.xdata)), int(round(ev.ydata)))
            if self.pending is None:
                self.pending = pt
            else:
                self._clear_preview()
                p1, self.pending = self.pending, None
                if p1 != pt:
                    self.add_profile(p1, pt)
            return
        for i, ax in enumerate(self.prof_axes[:len(self.profiles)]):
            if ev.inaxes is ax and ev.xdata is not None:
                self._measure_click(i, ev.xdata, ev.ydata)
                return

    def _measure_click(self, i, x, y):
        p, ax = self.profiles[i], self.prof_axes[i]
        if i not in self.meas_first:
            self.meas_first[i] = (x, y)
            ax.plot(x, y, "+", color=p["colour"], ms=10)
        else:
            (x0, y0) = self.meas_first.pop(i)
            if abs(y - y0) / (self.zlim[1] - self.zlim[0]) >= abs(x - x0) / (p["dist"][-1] + 1e-9):
                xm = (x + x0) / 2                       # vertical: depth / height difference
                ax.annotate("", xy=(xm, y), xytext=(xm, y0), arrowprops=dict(arrowstyle="<->", color="white"))
                ax.text(xm, (y + y0) / 2, f"  {abs(y - y0):.2f} {self.zunit}", color="white", fontsize=8,
                        va="center", family="monospace")
            else:
                ym = (y + y0) / 2                       # horizontal: width
                ax.annotate("", xy=(x, ym), xytext=(x0, ym), arrowprops=dict(arrowstyle="<->", color="white"))
                ax.text((x + x0) / 2, ym, f"{abs(x - x0):.2f} {self.xunit}\n", color="white", fontsize=8,
                        ha="center", va="bottom", family="monospace")
        self.fig.canvas.draw_idle()

    def on_move(self, ev):
        self._clear_cross()
        if ev.inaxes is self.ax_img and self.pending is not None and ev.xdata is not None:
            self._clear_preview()
            col = COLOURS[len(self.profiles) % len(COLOURS)]
            self.preview, = self.ax_img.plot([self.pending[0], ev.xdata], [self.pending[1], ev.ydata],
                                             color=col, lw=0.9, ls="--", alpha=0.8)
        for i, ax in enumerate(self.prof_axes[:len(self.profiles)]):
            if ev.inaxes is ax and ev.xdata is not None:
                p = self.profiles[i]
                k = int(np.argmin(np.abs(p["dist"] - ev.xdata)))
                x, y = p["xs"][k], p["ys"][k]
                h, w = self.photo.shape
                self.cross = [self.ax_img.plot([0, w - 1], [y, y], color=p["colour"], lw=0.7, ls="--", alpha=0.6)[0],
                              self.ax_img.plot([x, x], [0, h - 1], color=p["colour"], lw=0.7, ls="--", alpha=0.6)[0],
                              self.ax_img.plot(x, y, "o", color=p["colour"], ms=5)[0]]
                break
        self.fig.canvas.draw_idle()

    def on_key(self, ev):
        if ev.key == "u" and self.profiles:
            self.undo()
        elif ev.key == "c":
            while self.profiles:
                self.undo()
        elif ev.key == "s":
            self.save()

    # ------------------------------------------------------------------ actions
    def undo(self):
        p = self.profiles.pop()
        for a in p["artists"]:
            a.remove()
        i = len(self.profiles)
        self.prof_axes[i].cla(); self.prof_axes[i].set_visible(False)
        self.meas_first.pop(i, None)
        self.fig.canvas.draw_idle()

    def save(self, stem="crack_profiles"):
        self.fig.savefig(f"{stem}.png", dpi=150, facecolor=BG, bbox_inches="tight")
        with open(f"{stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["profile", f"distance_{self.xunit}", f"value_{self.zunit}", "x_px", "y_px"])
            for i, p in enumerate(self.profiles, 1):
                for d, z, x, y in zip(p["dist"], p["prof"], p["xs"], p["ys"]):
                    w.writerow([i, round(float(d), 4), round(float(z), 4), int(x), int(y)])
        print(f"Saved {Path(stem).resolve()}.png and .csv ({len(self.profiles)} profiles)")

    def show(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Interactive 3D crack profile tool")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--azimuth", nargs="+", type=float, default=[90, 0, 270, 180])
    ap.add_argument("--elevation", type=float, default=45.0)
    ap.add_argument("--px-mm", type=float, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    res, imgs, _ = run(a.images, a.azimuth, a.elevation, a.px_mm, a.scale)
    print(res.summary())
    ProfileTool(res, imgs.mean(0) ** (1 / 2.2), title=Path(a.images[0]).parent.name or "photos").show()


if __name__ == "__main__":
    main()
