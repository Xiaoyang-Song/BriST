"""Generate the synthetic BriST dataset.

python -m synthetic.generate --out data/synthetic_v1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np

from . import latent as LT
from .annotate import describe, measure
from .ct import LABELS, render_ct
from .spectra import LINES, N_CH, WL, render_spectra

LAYER_KEYS = ["e", "T", "s_lof", "s_kh", "vapor", "depth", "instab", "scan_angle"]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_v1")
    ap.add_argument("--n-builds", type=int, default=10)
    ap.add_argument("--parts-per-build", type=int, default=10)
    ap.add_argument("--regions-per-part", type=int, default=20)
    ap.add_argument("--annot-frac", type=float, default=0.10)
    ap.add_argument("--split-builds", type=int, nargs=3, default=[7, 1, 2], help="train/val/test builds")
    ap.add_argument("--annotator-probs", type=float, nargs=3, default=[0.45, 0.40, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    N = args.n_builds * args.parts_per_build * args.regions_per_part
    assert sum(args.split_builds) == args.n_builds

    perm = rng.permutation(args.n_builds)
    a, b = args.split_builds[0], args.split_builds[0] + args.split_builds[1]
    split_of = {int(x): s for s, grp in (("train", perm[:a]), ("val", perm[a:b]), ("test", perm[b:])) for x in grp}

    builds = [LT.sample_build(rng, i) for i in range(args.n_builds)]
    parts = []
    for bd in builds:
        for p in range(args.parts_per_build):
            pc = LT.sample_part(rng, bd, len(parts))
            pc["annotator"] = "ABC"[rng.choice(3, p=np.asarray(args.annotator_probs) / sum(args.annotator_probs))]
            parts.append(pc)

    h5 = h5py.File(os.path.join(args.out, "dataset.h5"), "w")
    h5.attrs["description"] = "Synthetic LPBF Ti-6Al-4V: in situ OES spectra (X), CT (U), expert text (Y)"
    h5.create_dataset("wavelength_nm", data=WL.astype(np.float32))
    dX = h5.create_dataset("spectra", (N, LT.N_LAYERS, N_CH), np.float32, chunks=(1, LT.N_LAYERS, N_CH),
                           compression="gzip", compression_opts=4)
    dU = h5.create_dataset("ct", (N, LT.NZ, LT.NY, LT.NX), np.uint8, chunks=(1, LT.NZ, LT.NY, LT.NX),
                           compression="gzip", compression_opts=4)
    dL = h5.create_dataset("ct_label", (N, LT.NZ, LT.NY, LT.NX), np.uint8, chunks=(1, LT.NZ, LT.NY, LT.NX),
                           compression="gzip", compression_opts=4)
    lat = {k: np.zeros((N, LT.N_LAYERS), np.float32) for k in LAYER_KEYS}
    h5["ct"].attrs["voxel_um"] = LT.VOXEL_UM
    h5["ct"].attrs["axes"] = "z(build, 0=bottom), y, x"
    h5["ct_label"].attrs["labels"] = json.dumps(LABELS)

    meta, texts = [], []
    t0 = time.time()
    i = 0
    for pc in parts:
        bd = builds[pc["build"]]
        for r in range(args.regions_per_part):
            reg = LT.sample_region(rng, bd, pc, r)
            defs = LT.sample_defects(rng, bd, reg)
            X = render_spectra(rng, bd, reg)
            U, lab = render_ct(rng, pc, defs)
            att = measure(lab)
            dX[i], dU[i], dL[i] = X, U, lab
            for k in LAYER_KEYS:
                lat[k][i] = reg[k]
            meta.append(dict(
                idx=i, sample_id=f"B{pc['build']:02d}_P{pc['part']:03d}_R{r:02d}",
                build=pc["build"], part=pc["part"], region=r, split=split_of[pc["build"]],
                doe_regime=pc["doe_regime"], power_W=pc["power_W"], speed_mm_s=pc["speed_mm_s"],
                ved_nom=pc["ved_nom"], z_frac=reg["z_frac"], events=reg["events"],
                moisture=bd["moisture"], oxygen=bd["oxygen"], gain=bd["gain"],
                lens_fouling=bd["lens_fouling"], wl_shift=bd["wl_shift"],
                ct_noise=pc["ct_noise"], ct_bh=pc["ct_bh"], ct_ring=pc["ct_ring"],
                n_true_defects={k: sum(d["kind"] == k for d in defs) for k in ("keyhole", "lof", "gas")},
                attributes=att,
            ))
            texts.append(dict(idx=i, sample_id=meta[-1]["sample_id"], annotator=pc["annotator"],
                              text=describe(att, rng, pc["annotator"])))
            i += 1
        if (pc["part"] + 1) % 10 == 0:
            print(f"  parts {pc['part'] + 1}/{len(parts)}  samples {i}/{N}  {time.time() - t0:.0f}s", flush=True)

    g = h5.create_group("latent")
    for k, v in lat.items():
        g.create_dataset(k, data=v)
    g.attrs["note"] = "per-layer hidden process state (oracle; never use as model input)"

    # expert-annotated subset: fraction of the train+val pool
    pool = np.array([m["idx"] for m in meta if m["split"] in ("train", "val")])
    n_ann = int(round(args.annot_frac * len(pool)))
    ann = set(rng.choice(pool, n_ann, replace=False).tolist())
    for m, t in zip(meta, texts):
        m["annotated"] = m["idx"] in ann
        t["split"] = m["split"]
        t["annotated"] = m["annotated"]

    split_arr = np.array([m["split"] for m in meta], dtype="S5")
    h5.create_dataset("split", data=split_arr)
    h5.create_dataset("annotated", data=np.array([m["annotated"] for m in meta]))
    h5.close()

    with open(os.path.join(args.out, "metadata.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(_jsonable(m)) + "\n")
    with open(os.path.join(args.out, "annotations.jsonl"), "w") as f:       # what the learner may use
        for t in texts:
            if t["annotated"]:
                f.write(json.dumps(t) + "\n")
    with open(os.path.join(args.out, "test_references.jsonl"), "w") as f:     # evaluation only
        for t in texts:
            if t["split"] == "test":
                f.write(json.dumps(t) + "\n")
    with open(os.path.join(args.out, "annotations_oracle.jsonl"), "w") as f:  # all, for ratio ablations
        for t in texts:
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), builds=_jsonable(builds), parts=_jsonable(parts),
                       split_of_build=split_of, lines=LINES, regimes=LT.REGIMES,
                       geometry=dict(n_layers=LT.N_LAYERS, layer_um=LT.LAYER_UM, voxel_um=LT.VOXEL_UM,
                                     shape_zyx=[LT.NZ, LT.NY, LT.NX])), f, indent=1)
    print(f"done: {N} samples, {n_ann} annotated train/val ({n_ann / len(pool):.0%}), {sum(m['split'] == 'test' for m in meta)} test refs,"
          f" {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
