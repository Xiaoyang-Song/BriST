"""CT-observable quality attributes and expert-style textual annotations (Y).

Attributes are measured from the CT ground-truth label volume only (defects
below a detectability threshold are ignored), so Y depends on the process
only through what CT shows (Y independent of X given U, by construction).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .latent import N_LAYERS, NX, NY, NZ, VOXEL_UM, VOX_PER_LAYER

MIN_VOX = 4                      # CT detectability (~20 um equivalent diameter)
STRUCT = np.ones((3, 3, 3), bool)
TYPE_IDS = {"keyhole": 1, "lack_of_fusion": 2, "gas": 3}
SEVERITY = ["none", "minor", "moderate", "severe"]


def _eqd(nvox):
    return 2 * (3 * nvox * VOXEL_UM ** 3 / (4 * np.pi)) ** (1 / 3)


def measure(label):
    """Return a dict of CT-observable attributes."""
    att = dict(types={})
    void_vox = 0
    all_z, all_xy, all_w = [], [], []
    for name, tid in TYPE_IDS.items():
        mask = label == tid
        if name == "lack_of_fusion":            # trapped powder belongs to the LoF defect
            mask_c = mask | (label == 4)
        else:
            mask_c = mask
        lab, n = ndimage.label(mask_c, STRUCT)
        comps = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            m = lab[sl] == i
            vox = int((m & mask[sl]).sum())
            if vox < MIN_VOX:
                continue
            idx = np.argwhere(m) + np.array([s.start for s in sl])
            ctr = idx.mean(0)
            cov = np.cov(idx.T) if len(idx) > 2 else np.zeros((3, 3))
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            length = float(2 * np.sqrt(5 * max(ev[0], 0)) * VOXEL_UM + VOXEL_UM)
            height = float((idx[:, 0].max() - idx[:, 0].min() + 1) * VOXEL_UM)
            comps.append(dict(vox=vox, eqd=float(_eqd(vox)), length=length, height=height,
                              z=float(ctr[0]), y=float(ctr[1]), x=float(ctr[2]),
                              has_powder=bool((label[sl][m] == 4).any())))
            void_vox += vox
            all_z.append(ctr[0]); all_xy.append(ctr[1:]); all_w.append(vox)
        if comps:
            att["types"][name] = dict(
                count=len(comps),
                volume_um3=float(sum(c["vox"] for c in comps) * VOXEL_UM ** 3),
                eqd_min=min(c["eqd"] for c in comps), eqd_max=max(c["eqd"] for c in comps),
                eqd_mean=float(np.mean([c["eqd"] for c in comps])),
                length_max=max(c["length"] for c in comps),
                height_max=max(c["height"] for c in comps),
                with_powder=int(sum(c["has_powder"] for c in comps)),
                layers=sorted({int(c["z"] // VOX_PER_LAYER) for c in comps}),
            )
    att["void_fraction_pct"] = float(100.0 * void_vox / (NX * NY * NZ))
    att["n_defects"] = int(sum(t["count"] for t in att["types"].values()))
    att["max_size_um"] = float(max([max(t["eqd_max"], t["length_max"] if k == "lack_of_fusion" else 0)
                                    for k, t in att["types"].items()] or [0.0]))

    # dominant type by void volume
    att["dominant"] = (max(att["types"], key=lambda k: att["types"][k]["volume_um3"])
                       if att["types"] else "none")

    # severity (porosity + critical size, as in common AM acceptance practice)
    vf, ms = att["void_fraction_pct"], att["max_size_um"]
    if att["n_defects"] == 0 or (vf < 0.01 and ms < 30):
        sev = "none"
    elif vf < 0.1 and ms < 100:
        sev = "minor"
    elif vf < 1.0 and ms < 300:
        sev = "moderate"
    else:
        sev = "severe"
    att["severity"] = sev

    # vertical distribution (layer index 0 = bottom of region)
    if all_w:
        w = np.asarray(all_w, float)
        zl = np.asarray(all_z) / VOX_PER_LAYER
        thirds = np.array([w[zl < N_LAYERS / 3].sum(), w[(zl >= N_LAYERS / 3) & (zl < 2 * N_LAYERS / 3)].sum(),
                           w[zl >= 2 * N_LAYERS / 3].sum()]) / w.sum()
        att["z_share"] = thirds.tolist()
        lay = np.bincount(zl.astype(int).clip(0, N_LAYERS - 1), weights=w, minlength=N_LAYERS)
        cum = lay / lay.sum()
        # smallest window of consecutive layers holding >=75% of void volume
        best = (N_LAYERS, 0, N_LAYERS - 1)
        for a in range(N_LAYERS):
            s = 0.0
            for b in range(a, N_LAYERS):
                s += cum[b]
                if s >= 0.75:
                    if b - a + 1 < best[0]:
                        best = (b - a + 1, a, b)
                    break
        att["z_window"] = [int(best[1]), int(best[2])]
        att["z_localized"] = bool(best[0] <= 4)
        xy = np.asarray(all_xy)
        cxy = (w[:, None] * xy).sum(0) / w.sum()
        spread = np.sqrt((w * ((xy - cxy) ** 2).sum(1)).sum() / w.sum()) if len(w) > 1 else 0.0
        att["xy_centroid"] = cxy.tolist()           # (y, x) in voxels
        att["xy_clustered"] = bool(len(w) >= 3 and spread < 16)
    else:
        att.update(z_share=[0, 0, 0], z_window=None, z_localized=False,
                   xy_centroid=None, xy_clustered=False)
    return att


# ---------------------------------------------------------------- text
def _vf(vf):
    return f"{vf:.2f}%" if vf >= 0.01 else "<0.01%"


def _r5(x):
    return int(5 * round(x / 5))


def _layers_phrase(rng, a, b):
    um0, um1 = a * 30, (b + 1) * 30
    if a == b:
        return rng.choice([f"layer {a + 1}", f"a single layer (layer {a + 1}, ~{um0}-{um1} um above the region base)"])
    return rng.choice([f"layers {a + 1}-{b + 1}", f"layers {a + 1}-{b + 1} (~{um0}-{um1} um above the region base)",
                       f"a narrow band at layers {a + 1}-{b + 1}"])


def _z_phrase(rng, att):
    if att["z_localized"]:
        return "concentrated in " + _layers_phrase(rng, *att["z_window"])
    zs = np.asarray(att["z_share"])
    if zs.max() > 0.6:
        return rng.choice(["mostly in the ", "predominantly in the "]) + ["lower", "middle", "upper"][int(zs.argmax())] + " third of the region"
    return rng.choice(["distributed throughout the build height", "spread over the full region height",
                       "scattered with no clear vertical concentration"])


def _xy_phrase(rng, att):
    if not att["xy_clustered"]:
        return None
    y, x = att["xy_centroid"]
    dx, dy = x - NX / 2, y - NY / 2
    if max(abs(dx), abs(dy)) < 8:
        side = "near the centre of the cross-section"
    elif abs(dx) >= abs(dy):
        side = "toward the +x edge" if dx > 0 else "toward the -x edge"
    else:
        side = "toward the +y edge" if dy > 0 else "toward the -y edge"
    return rng.choice([f"laterally clustered {side}", f"in plan view they cluster {side}"])


def _type_sentence(rng, name, t):
    n = t["count"]
    if name == "keyhole":
        rng_d = f"{_r5(t['eqd_min'])}-{_r5(t['eqd_max'])} um" if n > 1 else f"~{_r5(t['eqd_max'])} um"
        s = rng.choice([
            f"{n} near-spherical keyhole pore{'s' if n > 1 else ''} ({rng_d} equivalent diameter)",
            f"{n} rounded keyhole-type pore{'s' if n > 1 else ''} of {rng_d}",
        ])
        s += rng.choice([", located below the layer interfaces at roughly melt-pool depth",
                         ", sitting deep in the melt tracks", ""])
        return s
    if name == "lack_of_fusion":
        s = rng.choice([
            f"{n} irregular lack-of-fusion void{'s' if n > 1 else ''} up to {_r5(t['length_max'])} um long",
            f"{n} flat, irregularly shaped unfused region{'s' if n > 1 else ''} (longest ~{_r5(t['length_max'])} um)",
        ])
        s += rng.choice([" lying along the layer interfaces and elongated in the scan direction",
                         " aligned with the inter-layer boundaries", ""])
        if t["with_powder"]:
            s += rng.choice([", some containing unmelted powder particles",
                             ", with trapped unmelted powder visible inside"])
        return s
    # gas
    s = rng.choice([
        f"{n} small spherical gas pore{'s' if n > 1 else ''} (max {_r5(t['eqd_max'])} um)",
        f"{'isolated' if n < 4 else 'scattered'} fine gas porosity ({n} pore{'s' if n > 1 else ''} up to {_r5(t['eqd_max'])} um)",
    ])
    return s + rng.choice([", randomly dispersed", ""])


MECH = {
    "keyhole": ["consistent with keyhole-mode melting at excessive energy input",
                "indicative of an unstable keyhole / excessive energy density"],
    "lack_of_fusion": ["consistent with insufficient energy input / inadequate melt-pool overlap",
                       "indicative of incomplete fusion between layers"],
    "gas": ["typical of gas entrapped from the feedstock powder",
            "likely related to entrapped gas in the powder"],
}


def describe(att, rng, style):
    """Expert-style description. style: 'A' detailed, 'B' concise, 'C' structured."""
    sev, types = att["severity"], att["types"]
    vf = att["void_fraction_pct"]
    order = sorted(types, key=lambda k: -types[k]["volume_um3"])

    if style == "C":
        if not types:
            return "Defects: none detected | Severity: none | Void fraction: <0.01%"
        parts = [f"Defects: {', '.join(k.replace('_', ' ') for k in order)}",
                 f"Severity: {sev}",
                 f"Count: " + ", ".join(f"{k.replace('_', ' ')} {types[k]['count']}" for k in order),
                 f"Max size: {_r5(att['max_size_um'])} um",
                 f"Location: {_z_phrase(rng, att) if sev != 'none' else 'isolated'}",
                 f"Void fraction: {_vf(vf)}"]
        xy = _xy_phrase(rng, att) if sev != "none" else None
        if xy:
            parts.insert(5, f"Lateral: {xy}")
        return " | ".join(parts)

    if not types:
        return rng.choice([
            "No defects detected above the CT resolution limit; the region is fully dense.",
            "Region appears fully dense. No porosity or lack-of-fusion features are resolved.",
            "Dense material with no detectable porosity.",
        ])

    lead = {
        "minor": ["Minor porosity.", "Largely dense region with minor porosity.", "Low defect content."],
        "moderate": ["Moderate porosity.", "Region shows moderate porosity.", "A moderate defect population is present."],
        "severe": ["Severe porosity.", "Region is heavily defected.", "Severe defect content; region would likely fail acceptance."],
        "none": ["Essentially dense; only negligible porosity."],
    }[sev]
    dom = order[0]

    if style == "B":
        s = rng.choice(lead) + " "
        s += "Dominated by " + {"keyhole": "keyhole pores", "lack_of_fusion": "lack-of-fusion voids",
                                "gas": "small gas pores"}[dom]
        s += (", " + _z_phrase(rng, att) + ".") if sev != "none" else "."
        if len(order) > 1:
            s += " Also " + " and ".join({"keyhole": "some keyhole pores", "lack_of_fusion": "some lack-of-fusion",
                                          "gas": "minor gas porosity"}[k] for k in order[1:]) + "."
        return s

    # style A: detailed
    sents = [rng.choice(lead)]
    body = [_type_sentence(rng, k, types[k]) for k in order]
    sents.append("CT shows " + "; ".join(body) + ".")
    if sev != "none":
        loc = "Defects are " + _z_phrase(rng, att)
        xy = _xy_phrase(rng, att)
        if xy:
            loc += "; " + xy
        sents.append(loc + ".")
    sents.append(f"Estimated void fraction {_vf(vf)}.")
    if dom in MECH and (sev != "minor" or rng.random() < 0.5):
        sents.append(rng.choice(MECH[dom]).capitalize() + ".")
    return " ".join(sents)
