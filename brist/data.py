"""Data access for the synthetic BriST dataset: spectra, CT renderings, texts, labels."""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DEFECT_SEVERITIES = ("moderate", "severe")     # binary alarm target
SEV_LEVELS = ["none", "minor", "moderate", "severe"]


class SynthData:
    def __init__(self, root="data/synthetic_v1"):
        self.root = root
        self.meta = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl"))]
        self.text_all = {}
        for l in open(os.path.join(root, "annotations_oracle.jsonl")):
            t = json.loads(l)
            self.text_all[t["idx"]] = t
        self.N = len(self.meta)
        self.split = np.array([m["split"] for m in self.meta])
        self.annotated = np.array([m["annotated"] for m in self.meta])
        self.severity = np.array([m["attributes"]["severity"] for m in self.meta])
        self.defective = np.isin(self.severity, DEFECT_SEVERITIES).astype(np.int64)

    def idx(self, split):
        return np.where(self.split == split)[0]

    def annotated_idx(self, split="train", ratio=None, seed=0):
        """Expert-annotated subset. ratio=None -> the default 10% set shipped with the data;
        otherwise a nested random subset of the split at the requested ratio (for ablations)."""
        pool = self.idx(split)
        if ratio is None:
            return pool[self.annotated[pool]]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(pool)
        return np.sort(perm[: max(1, int(round(ratio * len(pool))))])

    def h5(self):
        return h5py.File(os.path.join(self.root, "dataset.h5"), "r")


# ------------------------------------------------------------------ spectra
def load_spectra(data: SynthData):
    with data.h5() as f:
        X = f["spectra"][:]
    return np.log(np.clip(X, 1.0, None)).astype(np.float32)


def standardize(X, train_idx):
    mu = X[train_idx].mean((0, 1), keepdims=True)
    sd = X[train_idx].std((0, 1), keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------ CT -> 2D image for the VLM
def ct_to_image(ct, size=512):
    """Render a CT volume (z,y,x uint8) as a 2x2 panel of minimum-intensity projections:
       [x-z MinIP | y-z MinIP]
       [x-y MinIP | x-y slice at the darkest layer]
    with build direction up and layer tick marks (1..16) on the left of the side views."""
    v = ndimage.gaussian_filter(ct.astype(np.float32), 0.8)
    nz, ny, nx = v.shape
    xz = v.min(1)[::-1]            # (z, x) with z up
    yz = v.min(2)[::-1]            # (z, y)
    xy = v.min(0)                  # (y, x)
    zbest = int(np.argmin(np.percentile(v.reshape(nz, -1), 1, axis=1)))
    sl = v[zbest]
    half = size // 2
    canvas = Image.new("L", (size, size), 0)

    def put(arr, x0, y0, side_view):
        a = np.clip((arr - 30) / (225 - 30) * 255, 0, 255).astype(np.uint8)
        h, w = a.shape
        pad_left = 22 if side_view else 0
        im = Image.fromarray(a).resize((half - pad_left - 4, int((half - pad_left - 4) * h / w)), Image.BILINEAR)
        oy = y0 + (half - im.size[1]) // 2
        canvas.paste(im, (x0 + pad_left, oy))
        if side_view:   # layer ticks: 16 layers of 3 voxels
            d = ImageDraw.Draw(canvas)
            ph = im.size[1]
            for l in range(16):
                yc = oy + ph - (l + 0.5) * ph / 16
                if l % 3 == 0 or l == 15:
                    d.text((x0 + 2, yc - 5), str(l + 1), fill=255)
                d.line([(x0 + pad_left - 4, yc), (x0 + pad_left - 1, yc)], fill=200)
    put(xz, 0, 0, True)
    put(yz, half, 0, True)
    put(xy, 0, half, False)
    put(sl, half, half, False)
    return canvas.convert("RGB")
