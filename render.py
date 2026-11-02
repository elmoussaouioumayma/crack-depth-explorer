"""Synthetic cracked concrete surfaces with known 3D shape, rendered under several lights.

Each scene is a height map in millimetres: a rough concrete surface with air voids
(round pits) and a branching crack network whose width and depth vary along each
crack. The scene is rendered as it would look to a camera pointing straight down,
lit one after another from several directions, including cast shadows inside the
cracks, a mottled surface colour (albedo), sensor noise and phone-like gamma.
Because the true height map is known, every measurement can be checked.

    python render.py                      # writes demo/scene_N.png ... and demo/truth.npz
    python render.py --seed 3 --out my_scene
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from photostereo import Lights


PROFILE_EXP = 1.3      # crack cross-section: depth * (1 - r)^1.3, between a V and a U shape


def half_depth_width(opening_mm):
    """Width at half depth of the crack cross-section used here (for comparison with measurements)."""
    return opening_mm * (1 - 0.5 ** (1 / PROFILE_EXP))


def _smooth_noise(shape, sigma, rng):
    n = gaussian_filter(rng.normal(0, 1, shape), sigma)
    return n / (n.std() + 1e-9)


def _crack_paths(h, w, rng, n_main=3):
    """Random-walk crack centre lines (main cracks plus branches), as lists of (x, y) points."""
    paths = []
    for _ in range(n_main):
        side = rng.integers(4)
        x, y = [(rng.uniform(0, w), 0), (w, rng.uniform(0, h)), (rng.uniform(0, w), h), (0, rng.uniform(0, h))][side]
        ang = np.arctan2(h / 2 - y, w / 2 - x) + rng.normal(0, 0.4)
        pts = []
        while -20 <= x <= w + 20 and -20 <= y <= h + 20 and len(pts) < 3000:
            pts.append((x, y))
            ang += rng.normal(0, 0.12)
            x, y = x + 2 * np.cos(ang), y + 2 * np.sin(ang)
        paths.append(np.array(pts))
        for _ in range(rng.integers(1, 4)):                       # branches
            k = rng.integers(len(pts) // 5, max(len(pts) // 5 + 1, 4 * len(pts) // 5))
            bx, by = pts[k]
            bang = ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            bpts = []
            for _ in range(int(rng.uniform(60, 220))):
                bpts.append((bx, by))
                bang += rng.normal(0, 0.15)
                bx, by = bx + 2 * np.cos(bang), by + 2 * np.sin(bang)
            paths.append(np.array(bpts))
    return paths


def make_scene(seed=0, h=480, w=640, px_mm=0.1):
    """Return (height_mm, albedo, crack_depth_mm, crack_width_mm, centre_mask).

    crack_depth_mm / crack_width_mm hold the true depth and width of the crack at
    each centre-line pixel (0 elsewhere)."""
    rng = np.random.default_rng(seed)
    # rough surface: broad undulation + fine texture (amplitudes in mm)
    height = 0.15 * _smooth_noise((h, w), 40, rng) + 0.03 * _smooth_noise((h, w), 3, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    # air voids: round, smooth dimples
    for _ in range(int(rng.uniform(15, 30))):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(4, 15)
        d = rng.uniform(0.3, 1.2)
        rr2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2
        height -= d * np.clip(1 - rr2, 0, None) ** 0.7

    # cracks: centre lines with depth and width varying smoothly along each crack
    centre = np.zeros((h, w), bool)
    cdepth = np.zeros((h, w))
    cwidth = np.zeros((h, w))
    for path in _crack_paths(h, w, rng):
        main = len(path) > 300
        t = np.linspace(0, len(path) - 1, 4 * len(path))           # resample to 0.5 px steps
        path = np.stack([np.interp(t, np.arange(len(path)), path[:, i]) for i in (0, 1)], 1)
        n = len(path)
        s = np.linspace(0, 1, n)
        wmax = rng.uniform(2.0, 4.5) if main else rng.uniform(0.8, 2.0)       # opening at the surface, mm
        dmax = wmax * (rng.uniform(0.3, 1.1) if main else rng.uniform(0.3, 0.9))  # depth / width ratio
        wave = 0.65 + 0.35 * np.sin(2 * np.pi * (s * rng.uniform(0.5, 2) + rng.uniform(0, 1)))
        taper = np.ones(n) if main else np.clip((1 - s) * 4, 0, 1)   # branches close towards their free end
        dep, wid = dmax * wave * taper, wmax * (0.6 + 0.4 * wave) * np.maximum(taper, 0.3)
        for (x, y), d_, w_ in zip(path, dep, wid):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and d_ > cdepth[yi, xi]:
                centre[yi, xi], cdepth[yi, xi], cwidth[yi, xi] = True, d_, w_
    centre &= cdepth > 0.05
    dist, (iy, ix) = distance_transform_edt(~centre, return_indices=True)
    d_near, w_near = cdepth[iy, ix], cwidth[iy, ix]
    r = dist * px_mm / np.maximum(w_near / 2, 1e-6)                  # 0 at the centre, 1 at the crack edge
    crack = d_near * np.clip(1 - r, 0, None) ** PROFILE_EXP
    height = gaussian_filter(height - crack, 0.7)

    # surface colour: mottled concrete, darker aggregate grains, slightly dirtier crack walls
    albedo = 0.55 + 0.08 * _smooth_noise((h, w), 6, rng) + 0.05 * _smooth_noise((h, w), 1, rng)
    grains = _smooth_noise((h, w), 4, rng) > 1.6
    albedo[grains] *= 0.75
    albedo *= 1 - 0.25 * (crack > 0.05)
    return height, np.clip(albedo, 0.15, 0.95), np.where(centre, cdepth, 0), np.where(centre, cwidth, 0), centre


def normals_from_height(height, px_mm):
    gy, gx = np.gradient(height, px_mm)
    gy = -gy                                   # image rows go down, our y axis goes up
    n = np.dstack([-gx, -gy, np.ones_like(height)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def cast_shadow(height, light, px_mm, max_mm=8.0):
    """True where the light is blocked by higher surface points (ray marching towards the light)."""
    h, w = height.shape
    lx, ly, lz = light
    horiz = np.hypot(lx, ly)
    if horiz < 1e-6:
        return np.zeros_like(height, bool)
    dx, dy = lx / horiz, -ly / horiz                        # image coordinates (rows go down)
    rise = lz / horiz * px_mm                               # height gain per pixel step along the ray
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    shadow = np.zeros((h, w), bool)
    for k in range(1, int(max_mm / px_mm)):
        hk = map_coordinates(height, [yy + k * dy, xx + k * dx], order=1, mode="nearest")
        shadow |= hk > height + k * rise
    return shadow


def render(height, albedo, light_dirs, px_mm, noise=0.01, ambient=0.03, gamma=True, seed=0):
    """8-bit images, one per light direction (Lambertian shading + cast shadows + noise)."""
    rng = np.random.default_rng(seed + 1000)
    n = normals_from_height(height, px_mm)
    out = []
    for L in np.atleast_2d(light_dirs):
        shade = np.clip(n @ L, 0, None) * ~cast_shadow(height, L, px_mm)
        img = albedo * (shade + ambient)
        img = img + rng.normal(0, noise, img.shape)
        img = np.clip(img / 0.9, 0, 1)
        if gamma:                                            # like a phone JPEG (sRGB)
            img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (1 / 2.4) - 0.055)
        out.append((img * 255).round().astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--px-mm", type=float, default=0.1, help="pixel size in mm")
    a = ap.parse_args()
    lights = Lights()
    height, albedo, cdepth, cwidth, centre = make_scene(a.seed, px_mm=a.px_mm)
    imgs = render(height, albedo, lights.directions(), a.px_mm, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in zip(lights.names, imgs):
        cv2.imwrite(str(out / f"scene_{name}.png"), img)
    np.savez_compressed(out / "truth.npz", height=height.astype(np.float32), crack_depth=cdepth.astype(np.float32),
                        crack_width=cwidth.astype(np.float32), albedo=albedo.astype(np.float32))
    (out / "lights.json").write_text(json.dumps({**asdict(lights), "px_mm": a.px_mm}, indent=2))
    print(f"Wrote {len(imgs)} images, truth.npz and lights.json to {out}/")


if __name__ == "__main__":
    main()
