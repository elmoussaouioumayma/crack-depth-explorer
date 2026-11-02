"""Create the demo scene and the figures used in the README.

    python make_figures.py
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from analyze import run
from photostereo import single_image_shading, to_linear
from profile_tool import ProfileTool
from render import make_scene, render
from report import cross_section, deepest_points, plot_report

PX = 0.1
FILES = [f"demo/scene_{n}.png" for n in "NESW"]


def main():
    subprocess.run([sys.executable, "render.py", "--seed", "7", "--out", "demo"], check=True)
    res, imgs, albedo = run(FILES, px_mm=PX)
    photo = imgs.mean(0) ** (1 / 2.2)

    # the four photos
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, f, name in zip(ax, FILES, ["light from the top", "from the right", "from the bottom", "from the left"]):
        a.imshow(cv2.imread(f, 0), cmap="gray")
        a.set_title(name)
        a.axis("off")
    fig.suptitle("Input: 4 photos, camera fixed, only the light moves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("images/photos.png", dpi=100)
    plt.close(fig)

    plot_report(res, photo, albedo, "images/report.png")
    plt.close("all")

    # comparison with the true surface and with the single-photo shortcut
    truth = np.load("demo/truth.npz")
    zt = truth["height"] - np.median(truth["height"])
    height, alb, *_ = make_scene(7, px_mm=PX)
    top = to_linear(render(height, alb, [0, 0, 1], PX, seed=7)[0])
    shade = single_image_shading(top)
    pts = deepest_points(res, n=3)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(top, cmap="gray")
    ax[0].set_title("One photo, light at the camera")
    ax[0].axis("off")
    for i, ((s, c, d), a) in enumerate(zip(pts, ax[1:])):
        t, z, xs, ys = cross_section(res.height, c, d, 40, PX)
        _, ztrue, _, _ = cross_section(zt, c, d, 40, PX)
        sh = map_coordinates(shade, [ys, xs], order=1)
        col = f"C{i}"
        ax[0].plot(xs, ys, lw=2.5, color=col)
        ax[0].text(xs[-1] + 4, ys[-1], str(i + 1), color=col, fontsize=11, fontweight="bold")
        a.plot(t, ztrue, "k", lw=2.5, label="true surface")
        a.plot(t, z, color="#d6452f", lw=2, label="photometric stereo")
        a.plot(t, -sh * 2.0, color="#9aa5b1", lw=2, ls="--", label="'darker = deeper' (x 2 mm)")
        a.set_title(f"Cross-section {i + 1}", color=f"C{i}", fontweight="bold")
        a.set_xlabel("mm")
        a.grid(alpha=0.3)
        if i == 0:
            a.set_ylabel("height (mm)")
            a.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig("images/truth_comparison.png", dpi=100)
    plt.close(fig)

    # interactive tool (scripted clicks)
    tool = ProfileTool(res, photo, title="demo scene")
    for p1, p2 in [((130, 40), (260, 90)), ((500, 120), (560, 260)), ((60, 330), (160, 430))]:
        tool.add_profile(p1, p2)
    d, z = tool.profiles[1]["dist"], tool.profiles[1]["prof"]
    k = int(np.argmin(z))
    tool._measure_click(1, d[k], z[k])
    tool._measure_click(1, d[k], 0.05)
    tool.fig.savefig("images/profile_tool.png", dpi=90, facecolor=tool.fig.get_facecolor(), bbox_inches="tight")
    print("Figures written to images/")


if __name__ == "__main__":
    main()
