"""Visualize circle/crack samples: (CT image, spectra) pairs, plus the expert text whenever the
region is annotated (triplets).

Examples
  # 12 random training regions; annotated ones are shown as triplets
  python -m dcca_sim.visualize --split train --n 12 --out figures/dataset/train_random.png
  # only annotated regions (triplets), cracks only
  python -m dcca_sim.visualize --only-annotated --kind crack --n 8 --out figures/dataset/annotated_cracks.png
  # specific regions, spectra at a different noise level
  python -m dcca_sim.visualize --idx 3 17 4021 --sigma 2.0 --out figures/dataset/picked.png
  # show the (normally hidden) oracle text for unannotated regions too, marked as such
  python -m dcca_sim.visualize --split test --n 8 --oracle-text --out figures/dataset/test_oracle.png

Each row: CT slice (3x3 location grid; 1 px = 10 um) | 16-d spectrum (noisy, and noise-free 2fc map
of V in grey) | expert annotation + CT-measured attributes (only if annotated or --oracle-text).
"""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_PORE, C_CRACK, C_CLEAN = "#2a78d6", "#eb6834", "#9a9893"


def load(data, sigma):
    meta = [json.loads(l) for l in open(os.path.join(data, "metadata.jsonl"))]
    imgs = np.load(os.path.join(data, "images.npy"))
    sp = np.load(os.path.join(data, f"spectra_sigma{sigma}.npy"))
    clean = None
    V = os.path.join(data, "V.npy")
    if os.path.exists(V):   # noise-free 2fc spectrum, reconstructed with the same (M, M2) as generation
        import torch
        from .dgp import make_spectra
        U = np.load(os.path.join(data, "U.npy"))
        clean, _, _, _ = make_spectra(U, 0.0)
    return meta, imgs, sp, clean


def pick(meta, args, rng):
    if args.idx:
        return list(args.idx)
    ids = [m["idx"] for m in meta
           if (args.split == "all" or m["split"] == args.split)
           and (not args.only_annotated or m["annotated"])
           and (args.kind == "all" or m["attributes"]["kind"] == args.kind)]
    if not ids:
        raise SystemExit("no regions match the filters")
    return sorted(rng.choice(ids, min(args.n, len(ids)), replace=False).tolist())


def attr_lines(a):
    if a["kind"] == "circle":
        it = "; ".join(f"Ø{t['diam_um']} µm @ {t['region']}" for t in a["items"])
        return [f"pores: {a['n']}  ({it})", f"overlap: {a['overlap']}   area: {a['area_pct']:.1f}%   severity: {a['severity']}"]
    it = "; ".join(f"{t['length_um']:.0f} µm {t['orient'].split(' (')[0]} @ {t['region']}" + (" [edge]" if t["edge"] else "")
                   for t in a["items"])
    return [f"cracks: {a['n']}  ({it})",
            f"intersecting: {a['intersect']}   total length: {a['total_length_um']:.0f} µm   severity: {a['severity']}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dcca_sim")
    ap.add_argument("--sigma", default="1.2")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--kind", default="all", choices=["all", "circle", "crack"])
    ap.add_argument("--only-annotated", action="store_true")
    ap.add_argument("--oracle-text", action="store_true", help="also show hidden text for unannotated regions")
    ap.add_argument("--idx", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--standardize", action="store_true",
                    help="show per-channel z-scored spectra (train statistics), i.e. what the models see")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/dataset/samples.png")
    args = ap.parse_args()

    meta, imgs, sp, clean = load(args.data, args.sigma)
    ids = pick(meta, args, np.random.default_rng(args.seed))
    if args.standardize:
        tr = [m["idx"] for m in meta if m["split"] == "train"]
        mu, sd = sp[tr].mean(0), sp[tr].std(0) + 1e-6
        sp = (sp - mu) / sd
        if clean is not None:
            clean = (clean - mu) / sd
    ylim = np.percentile(np.abs(sp), 99.5) * 1.1

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    fig = plt.figure(figsize=(15, 2.35 * len(ids)))
    H = 2.35 * len(ids)
    gs = fig.add_gridspec(len(ids), 3, width_ratios=[0.55, 1.25, 2.6], wspace=0.15, hspace=0.45,
                          top=1 - 0.75 / H, bottom=0.45 / H)
    for r, i in enumerate(ids):
        m = meta[i]
        a = m["attributes"]
        col = C_CRACK if a["kind"] == "crack" else C_PORE
        show_text = m["annotated"] or args.oracle_text

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(imgs[i], cmap="gray_r", vmin=0, vmax=1.6)
        for k in (32 / 3, 64 / 3):
            ax.axhline(k - 0.5, color=GRID, lw=0.6); ax.axvline(k - 0.5, color=GRID, lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        tag = "TRIPLET" if m["annotated"] else ("pair (+oracle text)" if args.oracle_text else "PAIR")
        ax.set_title(f"#{i}  {m['split']}  Y={m['Y']}\n{tag}", fontsize=8, color=INK, loc="left")

        ax = fig.add_subplot(gs[r, 1])
        x = np.arange(16)
        ax.bar(x, sp[i], color=col, width=0.7, label=f"X (σ={args.sigma})")
        if clean is not None:
            ax.plot(x, clean[i], color=C_CLEAN, lw=1.2, marker="o", ms=3, label="noise-free")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_ylim(-ylim, ylim); ax.set_xticks(x[::3])
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        if r == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        if r == len(ids) - 1:
            ax.set_xlabel("spectral channel" + (" (z-scored)" if args.standardize else " (raw 2fc output)"),
                          color=INK2)

        ax = fig.add_subplot(gs[r, 2]); ax.axis("off")
        if show_text:
            head = f"Expert annotation [annotator {m['annotator']}]" + ("" if m["annotated"] else "  — ORACLE (not in training set)")
            ax.text(0, 1.0, head, fontsize=8, fontweight="bold", color=INK if m["annotated"] else INK2,
                    va="top", transform=ax.transAxes)
            ax.text(0, 0.82, "\n".join(textwrap.wrap(m["text"], 120)), fontsize=8, color=INK, va="top",
                    transform=ax.transAxes)
            y0 = 0.36
        else:
            ax.text(0, 1.0, "No expert annotation (pair only)", fontsize=8, color=INK2, va="top",
                    transform=ax.transAxes)
            y0 = 0.78
        for k, line in enumerate(attr_lines(a)):
            ax.text(0, y0 - 0.15 * k, ("CT-measured: " if k == 0 else "             ") + line, fontsize=7.5,
                    color=INK2, va="top", transform=ax.transAxes, family="DejaVu Sans Mono")
    n_ann = sum(meta[i]["annotated"] for i in ids)
    fig.suptitle(f"Circle/crack simulation — {len(ids)} regions ({n_ann} annotated triplets, "
                 f"{len(ids) - n_ann} pairs); blue = pores (Y=0), orange = cracks (Y=1)",
                 x=0.01, y=1 - 0.12 / H, ha="left", fontsize=10.5, color=INK)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print("wrote", args.out, "regions:", ids)


if __name__ == "__main__":
    main()
