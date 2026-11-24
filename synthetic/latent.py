"""Shared latent process state for synthetic LPBF (Ti-6Al-4V) regions.

Both the in situ spectra (X) and the offline CT (U) are rendered from the
objects produced here, so the two modalities share one physical cause:
the per-layer effective energy input and the melt-pool physics it induces.

Hierarchy:  build -> part -> region -> layer
  build  : powder moisture, chamber oxygen, sensor gain/dark/wavelength shift,
           lens-fouling rate                              (shared + X nuisance)
  part   : nominal laser power / scan speed (process-map DOE), scan rotation,
           height in build, CT scan settings              (shared + U nuisance)
  region : 16 layers, local energy drift, power-dip / overheating events
  layer  : energy ratio e, melt-pool T, depth, vapor, keyhole / LoF propensity
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- geometry
N_LAYERS = 16
LAYER_UM = 30.0           # powder layer thickness
VOXEL_UM = 10.0           # CT voxel size
VOX_PER_LAYER = int(LAYER_UM / VOXEL_UM)
NX = NY = 64              # region footprint: 640 x 640 um
NZ = N_LAYERS * VOX_PER_LAYER   # 48 voxels = 480 um
REGION_XY_UM = NX * VOXEL_UM
REGION_Z_UM = NZ * VOXEL_UM

HATCH_MM = 0.10
LAYER_MM = LAYER_UM / 1000.0
VED_OPT = 70.0            # J/mm^3, centre of the Ti-6Al-4V process window
SCAN_ROT_DEG = 67.0       # inter-layer scan rotation

# DOE regimes for nominal volumetric energy density (J/mm^3)
REGIMES = {
    "lack_of_fusion": (0.25, (28.0, 50.0)),
    "nominal":        (0.45, (52.0, 90.0)),
    "keyhole":        (0.30, (105.0, 170.0)),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


# ---------------------------------------------------------------- sampling
def sample_build(rng, b):
    return dict(
        build=b,
        moisture=float(np.clip(rng.lognormal(0.0, 0.55), 0.3, 4.0)),   # powder humidity index
        oxygen=float(rng.uniform(0.2, 1.0)),                           # chamber O2 index (X only)
        gain=float(rng.normal(1.0, 0.08)),                             # spectrometer gain
        dark=float(rng.uniform(80, 140)),                              # dark offset (counts)
        lens_fouling=float(rng.uniform(0.0, 0.5)),                     # transmission loss over build
        wl_shift=float(rng.normal(0.0, 0.12)),                         # wavelength calibration (nm)
        stray=rng.normal(0, 1, 3).tolist(),                            # stray-light baseline coeffs
    )


def sample_part(rng, build, p):
    names = list(REGIMES)
    probs = np.array([REGIMES[k][0] for k in names])
    regime = names[rng.choice(len(names), p=probs / probs.sum())]
    lo, hi = REGIMES[regime][1]
    for _ in range(100):
        ved = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        P = float(rng.uniform(150, 370))                     # W
        v = P / (ved * HATCH_MM * LAYER_MM)                  # mm/s
        if 400 <= v <= 2000:
            break
    return dict(
        part=p, build=build["build"], doe_regime=regime,
        power_W=P, speed_mm_s=float(v), ved_nom=ved,
        scan_angle0=float(rng.uniform(0, 180)),
        z_base=float(rng.uniform(0.0, 0.6)),                 # part start height (fraction of build)
        # CT acquisition (per part scan)
        ct_noise=float(rng.uniform(0.03, 0.07)),
        ct_psf=float(rng.uniform(0.6, 0.9)),
        ct_bh=float(rng.uniform(0.0, 0.05)),
        ct_ring=float(rng.uniform(0.0, 0.03)),
        ct_axis=rng.uniform(-400, 1000, 2).tolist(),         # rotation axis (voxels) rel. to region
    )


def sample_region(rng, build, part, r):
    L = N_LAYERS
    z_frac = part["z_base"] + rng.uniform(0.0, 0.35)          # height of region in build
    heat = 1.0 + 0.10 * (z_frac - part["z_base"]) / 0.35     # heat accumulation with height

    # AR(1) layer-to-layer energy drift
    eps = np.zeros(L)
    for l in range(L):
        eps[l] = (0.6 * eps[l - 1] if l else 0.0) + rng.normal(0, 0.04)
    e = part["ved_nom"] / VED_OPT * heat * np.exp(eps)

    events = []
    mult = np.ones(L)
    if rng.random() < 0.12:     # laser power dip (e.g. optics/plume shielding)
        s, d = rng.integers(0, L), rng.integers(1, 4)
        m = rng.uniform(0.45, 0.75)
        mult[s:s + d] *= m
        events.append(dict(kind="power_dip", start=int(s), end=int(min(s + d, L) - 1), factor=float(m)))
    if rng.random() < 0.08:     # local overheating (short vectors / heat sink loss)
        s, d = rng.integers(0, L), rng.integers(2, 6)
        m = rng.uniform(1.4, 1.9)
        mult[s:s + d] *= m
        events.append(dict(kind="overheating", start=int(s), end=int(min(s + d, L) - 1), factor=float(m)))
    e = e * mult

    T = np.clip(2400 + 700 * np.log2(e) + rng.normal(0, 30, L), 1500, 3400)   # melt-pool peak T (K)
    s_lof = sigmoid((0.62 - e) / 0.05)
    s_kh = sigmoid((e - 1.70) / 0.08)
    vapor = 0.15 * softplus((e - 1.1) / 0.15)
    depth = 60 * e ** 0.9 + 80 * s_kh                                          # melt-pool depth (um)
    instab = 0.05 + 0.6 * s_kh + 0.4 * s_lof
    scan_angle = (part["scan_angle0"] + SCAN_ROT_DEG * np.arange(L)) % 180.0

    return dict(
        region=r, part=part["part"], build=build["build"], z_frac=float(z_frac),
        e=e, T=T, s_lof=s_lof, s_kh=s_kh, vapor=vapor, depth=depth,
        instab=instab, scan_angle=scan_angle, events=events,
    )


# ---------------------------------------------------------------- defects
def sample_defects(rng, build, region):
    """Return a list of defect primitives in um coordinates (x, y, z up)."""
    defs = []
    for l in range(N_LAYERS):
        e, skh, slof = region["e"][l], region["s_kh"][l], region["s_lof"][l]
        z_top = LAYER_UM * (l + 1)

        # keyhole pores: collapse of unstable keyhole at melt-pool bottom
        for _ in range(rng.poisson(0.8 * skh)):
            r = rng.uniform(18, 30) + 25 * skh * rng.uniform(0.5, 1.2)
            # may fall below the region (z<0): then it belongs to the region underneath
            zc = z_top - region["depth"][l] * rng.uniform(0.7, 1.0)
            defs.append(dict(kind="keyhole", layer=l,
                             c=[rng.uniform(0, REGION_XY_UM), rng.uniform(0, REGION_XY_UM), zc],
                             axes=[r, r, r * rng.uniform(1.0, 1.3)], angle=0.0))

        # lack of fusion: unfused gaps at layer interface, along scan tracks
        for _ in range(rng.poisson(2.5 * slof)):
            a = rng.uniform(100, 350) * (0.5 + slof) / 2
            b = rng.uniform(40, 90) / 2
            c = rng.uniform(16, 36) / 2
            zc = LAYER_UM * l + rng.normal(0, 5)
            ang = region["scan_angle"][l] + rng.normal(0, 8)
            cxy = [rng.uniform(0, REGION_XY_UM), rng.uniform(0, REGION_XY_UM)]
            k = rng.poisson(1.5 * (2 * a / 200))
            parts = []
            th = np.deg2rad(ang)
            for _ in range(k):
                u, w = rng.uniform(-0.6, 0.6) * a, rng.uniform(-0.5, 0.5) * b
                parts.append(dict(c=[cxy[0] + u * np.cos(th) - w * np.sin(th),
                                     cxy[1] + u * np.sin(th) + w * np.cos(th), zc],
                                  r=float(rng.uniform(8, 18))))
            defs.append(dict(kind="lof", layer=l, c=[cxy[0], cxy[1], zc], axes=[a, b, c],
                             angle=float(ang), particles=parts,
                             rough=dict(k=rng.normal(0, 2 * np.pi / 40, (4, 3)).tolist(),
                                        ph=rng.uniform(0, 2 * np.pi, 4).tolist())))

        # gas pores: entrapped gas from moist powder, size often near CT resolution
        lam = 0.5 * build["moisture"] * (0.6 + 0.4 * min(e, 1.5))
        for _ in range(rng.poisson(lam)):
            r = float(np.clip(rng.lognormal(np.log(8.0), 0.4), 3, 25))
            defs.append(dict(kind="gas", layer=l,
                             c=[rng.uniform(0, REGION_XY_UM), rng.uniform(0, REGION_XY_UM),
                                LAYER_UM * l + rng.uniform(0, LAYER_UM)],
                             axes=[r, r, r], angle=0.0))
    return defs
