"""Photometric stereo: 3D surface shape from photos taken with light from different sides.

The camera stays fixed and looks straight down; only the light moves. For a matte
surface, the brightness of a pixel is  albedo * (normal . light direction).  With
three or more known light directions, the surface normal and the albedo (surface
colour) of every pixel can be solved independently of each other, so a dark stain
is no longer mistaken for a deep hole. Integrating the normals gives the height map.

    normals, albedo = estimate_normals(images, light_dirs)
    height_mm = integrate(normals, px_mm)
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter


@dataclass
class Lights:
    azimuth_deg: tuple = (90.0, 0.0, 270.0, 180.0)   # light comes FROM: N (top), E (right), S, W
    elevation_deg: float = 45.0                      # angle above the surface
    names: tuple = ("N", "E", "S", "W")

    def directions(self, elevation_deg=None):
        """Unit vectors (x right, y up in the image, z towards the camera)."""
        el = np.deg2rad(self.elevation_deg if elevation_deg is None else elevation_deg)
        az = np.deg2rad(np.array(self.azimuth_deg))
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full(az.shape, np.sin(el))], 1)


def to_linear(img8, srgb=True):
    """8-bit image -> linear brightness in [0, 1] (phone JPEGs are sRGB-encoded)."""
    x = img8.astype(np.float64) / 255.0
    if not srgb:
        return x
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_images(paths, srgb=True, scale=1.0):
    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        imgs.append(to_linear(im, srgb))
    return np.stack(imgs)


def estimate_normals(images, light_dirs, shadow_ratio=0.25):
    """Per-pixel least squares for g = albedo * normal from I = L g.

    With four or more lights, the darkest observation of a pixel is dropped when it is
    much darker than the others (the pixel is then probably in a cast shadow for that
    light, which breaks the model); the remaining lights still determine the normal.

    images: (k, H, W) linear brightness; light_dirs: (k, 3) unit vectors (x right, y up, z to camera).
    Returns normals (H, W, 3) and albedo (H, W).
    """
    k, H, W = images.shape
    L = np.asarray(light_dirs, float)
    I = images.reshape(k, -1)
    g = np.linalg.lstsq(L, I, rcond=None)[0]                     # all lights
    if k >= 4:
        darkest = I.argmin(0)
        others = np.sort(I, 0)[1:].mean(0)
        shadowed = I.min(0) < shadow_ratio * others
        for j in range(k):                                       # solve again without light j
            sel = shadowed & (darkest == j)
            if sel.any():
                keep = [i for i in range(k) if i != j]
                g[:, sel] = np.linalg.lstsq(L[keep], I[keep][:, sel], rcond=None)[0]
    g = g.T.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=2)
    n = g / np.maximum(albedo[..., None], 1e-9)
    n[n[..., 2] < 0.05] = [0, 0, 1]                              # unreliable, e.g. black in every image
    return n, albedo


def integrate(normals, px_mm, max_slope=5.0):
    """Height map (mm) from normals: least-squares integration of the gradient field.

    Solves the Poisson equation lap(z) = div(p, q) with Neumann boundary conditions
    using the discrete cosine transform (Simchony, Chellappa & Shao 1990).
    """
    nz = np.maximum(normals[..., 2], 1.0 / max_slope)
    p = np.clip(-normals[..., 0] / nz, -max_slope, max_slope)        # dz/dx
    q = np.clip(normals[..., 1] / nz, -max_slope, max_slope)         # dz/d(row): rows go down, y goes up
    H, W = p.shape
    # slopes on the edges between neighbouring pixels (no edge beyond the border)
    pe = np.zeros((H, W))
    qe = np.zeros((H, W))
    pe[:, :-1] = 0.5 * (p[:, :-1] + p[:, 1:])
    qe[:-1, :] = 0.5 * (q[:-1, :] + q[1:, :])
    # divergence: backward differences of the edge slopes
    f = pe.copy()
    f[:, 1:] -= pe[:, :-1]
    f += qe
    f[1:, :] -= qe[:-1, :]
    F = dctn(f, type=2, norm="ortho")
    yy, xx = np.mgrid[0:H, 0:W]
    denom = (2 * np.cos(np.pi * xx / W) - 2) + (2 * np.cos(np.pi * yy / H) - 2)
    denom[0, 0] = 1.0
    Z = F / denom
    Z[0, 0] = 0.0
    z = idctn(Z, type=2, norm="ortho") * px_mm
    return z - np.median(z)


def remove_trend(height, px_mm, sigma_mm=None):
    """Remove a tilted plane (default) or, with sigma_mm, all shape broader than sigma_mm.

    With real photos the light is not infinitely far away and not perfectly even, which
    bends the integrated surface slightly. Cracks are narrow, so removing the broad
    shape keeps their depth intact.
    """
    H, W = height.shape
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(H * W)]
    coef = np.linalg.lstsq(A, height.ravel(), rcond=None)[0]
    z = height - (A @ coef).reshape(H, W)
    if sigma_mm:
        z = z - gaussian_filter(z, sigma_mm / px_mm)
    return z - np.median(z)


def single_image_shading(image, crack_mask=None):
    """The shortcut photometric stereo replaces: 'darker = deeper' from ONE image.

    Returns a darkness index in [0, 1] relative to the local surface brightness. It
    has no unit and mixes depth with surface colour, shadows and crack width.
    """
    ref = cv2.morphologyEx((image * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(float) / 255
    ref = gaussian_filter(ref, 3)
    idx = np.clip((ref - image) / np.maximum(ref, 1e-6), 0, 1)
    return idx if crack_mask is None else np.where(crack_mask, idx, 0)
