"""Offline X-ray CT volume (U) of each region, rendered from the defect list.

Pipeline: analytic defect geometry -> partial-volume occupancy -> attenuation
map with build texture -> PSF blur -> beam hardening (cupping) -> ring
artifacts -> noise -> 8-bit quantisation. A ground-truth label volume is
produced alongside (0 matrix, 1 keyhole, 2 lack of fusion, 3 gas, 4 unmelted
powder inside a LoF void).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .latent import NX, NY, NZ, VOXEL_UM, VOX_PER_LAYER

LABELS = {0: "matrix", 1: "keyhole", 2: "lack_of_fusion", 3: "gas", 4: "unmelted_powder"}
KIND2ID = {"keyhole": 1, "lof": 2, "gas": 3}

# voxel centres in um; arrays indexed [z, y, x]
_ZC = (np.arange(NZ) + 0.5) * VOXEL_UM
_YC = (np.arange(NY) + 0.5) * VOXEL_UM
_XC = (np.arange(NX) + 0.5) * VOXEL_UM


def _bbox(c, ext):
    lo = [max(int((c[i] - ext) // VOXEL_UM), 0) for i in range(3)]
    hi = [min(int((c[i] + ext) // VOXEL_UM) + 2, n) for i, n in enumerate((NX, NY, NZ))]
    return lo, hi


def _occupancy(d):
    """Signed distance (um, negative inside) -> partial-volume fraction."""
    return np.clip(0.5 - d / VOXEL_UM, 0.0, 1.0)


def _ellipsoid(c, axes, angle_deg, rough=None):
    ext = max(axes) + 2 * VOXEL_UM
    lo, hi = _bbox(c, ext)
    if any(h <= l for l, h in zip(lo, hi)):
        return None
    z, y, x = np.meshgrid(_ZC[lo[2]:hi[2]], _YC[lo[1]:hi[1]], _XC[lo[0]:hi[0]], indexing="ij")
    dx, dy, dz = x - c[0], y - c[1], z - c[2]
    th = np.deg2rad(angle_deg)
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    rho = np.sqrt((u / axes[0]) ** 2 + (v / axes[1]) ** 2 + (dz / axes[2]) ** 2)
    d = (rho - 1.0) * min(axes)
    if rough is not None:  # irregular boundary for lack-of-fusion voids
        k, ph = np.asarray(rough["k"]), np.asarray(rough["ph"])
        n = sum(np.sin(k[i, 0] * x + k[i, 1] * y + k[i, 2] * z + ph[i]) for i in range(len(ph)))
        d = d + 0.6 * min(axes) * n / np.sqrt(len(ph)) + (rho - 1.0) * 0.3 * axes[1]
    return (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0])), _occupancy(d)


def rasterize(defects):
    """Return void occupancy per class (3, NZ, NY, NX), particle occupancy, label volume."""
    occ = np.zeros((3, NZ, NY, NX), np.float32)
    part = np.zeros((NZ, NY, NX), np.float32)
    for df in defects:
        res = _ellipsoid(df["c"], df["axes"], df["angle"], df.get("rough"))
        if res is None:
            continue
        sl, o = res
        k = KIND2ID[df["kind"]] - 1
        occ[(k,) + sl] = np.maximum(occ[(k,) + sl], o)
        for p in df.get("particles", []):
            r2 = _ellipsoid(p["c"], [p["r"]] * 3, 0.0)
            if r2 is not None:
                s2, o2 = r2
                part[s2] = np.maximum(part[s2], o2)
    void = occ.max(0)
    lof_void = occ[1]
    part = part * (lof_void > 0.3)                 # powder only visible inside the gap
    void = void * (1 - part)
    label = np.where(void > 0.5, occ.argmax(0) + 1, 0).astype(np.uint8)
    label[(part > 0.5)] = 4
    return void, label


def render_ct(rng, part_cfg, defects):
    """Return (ct uint8 [NZ,NY,NX], label uint8 [NZ,NY,NX])."""
    void, label = rasterize(defects)
    mu = 1.0 - void

    # build texture: faint layer banding + low-frequency microstructure contrast
    zz = np.arange(NZ)[:, None, None]
    mu = mu + 0.012 * np.cos(2 * np.pi * zz / VOX_PER_LAYER) * (1 - void)
    tex = ndimage.gaussian_filter(rng.normal(0, 1, mu.shape), 3.0)
    mu = mu + 0.012 * tex / (tex.std() + 1e-8) * (1 - void)

    # scanner PSF
    mu = ndimage.gaussian_filter(mu, part_cfg["ct_psf"])

    # beam hardening cupping + ring artefacts around the rotation axis
    ax, ay = part_cfg["ct_axis"]
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    rr = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2)
    R = 1500.0
    mu = mu * (1 - part_cfg["ct_bh"] * (1 - np.clip(rr / R, 0, 1) ** 2))[None]
    ring = np.zeros_like(rr)
    for _ in range(rng.integers(2, 6)):
        r0 = rng.uniform(rr.min(), rr.max())
        ring += rng.choice([-1, 1]) * np.exp(-0.5 * ((rr - r0) / 0.6) ** 2)
    mu = mu + part_cfg["ct_ring"] * ring[None]

    mu = mu + rng.normal(0, part_cfg["ct_noise"], mu.shape)
    ct = np.clip((mu + 0.25) / 1.5 * 255, 0, 255).astype(np.uint8)
    return ct, label
