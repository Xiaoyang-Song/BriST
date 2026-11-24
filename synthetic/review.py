"""Review figures for a generated synthetic dataset.

python -m synthetic.review --data data/synthetic_v1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import textwrap

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .latent import VED_OPT, VOX_PER_LAYER

DOM_COL = {"none": "#9aa0a6", "gas": "#2a9d8f", "keyhole": "#e76f51", "lack_of_fusion": "#264653"}
REG_COL = {"lack_of_fusion": "#264653", "nominal": "#2a9d8f", "keyhole": "#e76f51"}


def band(wl, X, lo, hi):
    m = (wl >= lo) & (wl <= hi)
    return X[..., m].max(-1) - np.median(X[..., m], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_v1")
    args = ap.parse_args()
    out = os.path.join(args.data, "review")
    os.makedirs(out, exist_ok=True)
    M = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    T = [json.loads(l) for l in open(os.path.join(args.data, "annotations_oracle.jsonl"))]
    f = h5py.File(os.path.join(args.data, "dataset.h5"), "r")
    wl = f["wavelength_nm"][:]
    e = f["latent/e"][:]
    dom = np.array([m["attributes"]["dominant"] for m in M])
    vf = np.array([m["attributes"]["void_fraction_pct"] for m in M])

    # ---------- 1. process map
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    P = np.array([m["power_W"] for m in M]); v = np.array([m["speed_mm_s"] for m in M])
    for k, c in DOM_COL.items():
        s = dom == k
        ax[0].scatter(v[s], P[s], s=10, c=c, label=k, alpha=0.7)
    vv = np.linspace(400, 2000, 50)
    for ved in (40, 70, 110):
        ax[0].plot(vv, ved * 0.1 * 0.03 * vv, "k--", lw=0.7)
        ax[0].text(vv[-1], ved * 0.003 * vv[-1], f" {ved} J/mm³", fontsize=7, va="center")
    ax[0].set(xlabel="scan speed (mm/s)", ylabel="laser power (W)", ylim=(140, 380),
              title="Process map: part DOE, coloured by CT-dominant defect")
    ax[0].legend(fontsize=8, markerscale=2)
    em = e.mean(1)
    for k, c in DOM_COL.items():
        s = dom == k
        ax[1].scatter(em[s], np.maximum(vf[s], 1e-3), s=8, c=c, alpha=0.6, label=k)
    ax[1].set(xscale="log", yscale="log", xlabel="mean effective energy ratio  E/E_opt  (latent)",
              ylabel="CT void fraction (%)", title="Shared latent -> CT outcome")
    ax[1].axvline(0.62, c="k", lw=0.5, ls=":"); ax[1].axvline(1.7, c="k", lw=0.5, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_process_map.png"), dpi=130); plt.close(fig)

    # ---------- 2. spectra by regime
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3)
    a0 = fig.add_subplot(gs[0, :])
    picks = {}
    for k in ["lack_of_fusion", "none", "keyhole"]:
        cand = [i for i, m in enumerate(M) if m["attributes"]["dominant"] == k and not m["events"]]
        if k == "none":
            cand = [i for i in cand if M[i]["doe_regime"] == "nominal"]
        if cand:
            picks[k] = cand[0]
    lab = {"lack_of_fusion": "LoF region", "none": "dense (nominal)", "keyhole": "keyhole region"}
    col = {"lack_of_fusion": "#264653", "none": "#2a9d8f", "keyhole": "#e76f51"}
    for k, i in picks.items():
        X = f["spectra"][i].mean(0)
        a0.semilogy(wl, X, c=col[k], lw=0.8, label=f"{lab[k]}  (mean E/E_opt={e[i].mean():.2f})")
    for c, name in [(394.4, "Al I"), (396.15, ""), (437.9, "V I"), (453.3, "Ti I"), (498.2, "Ti I"),
                    (656.3, "H I"), (763.5, "Ar I"), (777.2, "O I"), (811.5, "Ar I")]:
        if name:
            a0.annotate(name, (c, a0.get_ylim()[1] if False else 1), xycoords=("data", "axes fraction"),
                        xytext=(0, -10), textcoords="offset points", ha="center", fontsize=7, color="0.3")
    a0.set(xlabel="wavelength (nm)", ylabel="counts (layer-averaged)", title="In situ emission spectra by regime")
    a0.legend(fontsize=8)
    for j, (lo, hi, t) in enumerate([(390, 400, "Al I doublet"), (495, 503, "Ti I multiplet"), (755, 815, "O I / Ar I")]):
        ax = fig.add_subplot(gs[1, j])
        m = (wl >= lo) & (wl <= hi)
        for k, i in picks.items():
            X = f["spectra"][i].mean(0)
            ax.plot(wl[m], X[m] / np.median(X[m]), c=col[k], lw=1)
        ax.set(title=t + " (normalised to local median)", xlabel="nm")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_spectra.png"), dpi=130); plt.close(fig)

    # ---------- 3. paired examples
    want = [("dense", lambda m: m["attributes"]["severity"] == "none"),
            ("lack of fusion", lambda m: m["attributes"]["dominant"] == "lack_of_fusion" and m["attributes"]["severity"] == "severe"),
            ("keyhole", lambda m: m["attributes"]["dominant"] == "keyhole" and m["attributes"]["severity"] in ("moderate", "severe")),
            ("power-dip event", lambda m: any(ev["kind"] == "power_dip" for ev in m["events"]) and m["doe_regime"] == "nominal"
             and m["attributes"]["z_localized"]),
            ("overheating event", lambda m: any(ev["kind"] == "overheating" for ev in m["events"]) and m["doe_regime"] == "nominal"
             and m["attributes"]["n_defects"] > 0)]
    rows, used = [], set()
    for name, fn in want:
        c = [i for i, m in enumerate(M) if fn(m) and i not in used]
        if c:
            rows.append((name, c[0])); used.add(c[0])
    fig, axs = plt.subplots(len(rows), 5, figsize=(17, 3.5 * len(rows)),
                            gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.75, 0.75, 1.9]))
    axs = np.atleast_2d(axs)
    for r, (name, i) in enumerate(rows):
        X = f["spectra"][i]; U = f["ct"][i]; L = f["ct_label"][i]
        ax = axs[r, 0]
        ax.imshow(np.log10(np.clip(X, 1, None)), aspect="auto", origin="lower", cmap="magma",
                  extent=[wl[0], wl[-1], -0.5, 15.5])
        ax.set(ylabel="layer", xlabel="nm", title=f"{name}: X (log counts)")
        ax2 = ax.twiny(); ax2.plot(e[i], np.arange(16), c="cyan", lw=1.5); ax2.set_xlabel("latent E/E_opt", color="c", fontsize=7)
        ax2.tick_params(labelsize=6, colors="c")
        vox = (L > 0) & (L < 4)
        yslice = int(np.argmax(vox.sum((0, 2)))) if vox.any() else 32
        zslice = int(np.argmax(vox.sum((1, 2)))) if vox.any() else 24
        axs[r, 1].imshow(U[:, yslice, :], cmap="gray", origin="lower", vmin=0, vmax=255, aspect="auto")
        axs[r, 1].set(title=f"U: x-z slice (y={yslice})", ylabel="z (vox, 10 um)")
        axs[r, 2].imshow(U[zslice], cmap="gray", vmin=0, vmax=255)
        axs[r, 2].set(title=f"U: x-y slice (z={zslice}, layer {zslice // VOX_PER_LAYER + 1})")
        proj = np.zeros(L.shape[::2][:1] + (L.shape[2],) + (3,))
        lab3 = np.zeros((L.shape[0], L.shape[2], 3))
        for tid, rgb in [(1, (0.9, 0.44, 0.32)), (2, (0.15, 0.27, 0.33)), (3, (0.16, 0.62, 0.56)), (4, (0.95, 0.8, 0.2))]:
            lab3[(L == tid).any(1)] = rgb
        axs[r, 3].imshow(np.where(lab3.sum(-1, keepdims=True) > 0, lab3, 1.0), origin="lower", aspect="auto")
        axs[r, 3].set(title="labels: x-z projection")
        axs[r, 4].axis("off")
        txt = T[i]["text"]
        a = M[i]["attributes"]
        axs[r, 4].text(0, 1, "\n".join(textwrap.wrap(f"Y [{T[i]['annotator']}]: {txt}", 70)), va="top", fontsize=8,
                       family="monospace")
        axs[r, 4].text(0, 0.02, f"{M[i]['sample_id']}  split={M[i]['split']}  regime={M[i]['doe_regime']}\n"
                       f"events={[(ev['kind'], ev['start'] + 1, ev['end'] + 1) for ev in M[i]['events']]}\n"
                       f"severity={a['severity']} void={a['void_fraction_pct']:.2f}%", fontsize=7, color="0.35")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_paired_examples.png"), dpi=110); plt.close(fig)

    # ---------- 4. shared latent: layer-level spectral features vs CT
    n = min(len(M), 600)
    X = f["spectra"][:n]
    L = f["ct_label"][:n]
    layer_void = np.stack([((L[:, 3 * l:3 * l + 3] > 0) & (L[:, 3 * l:3 * l + 3] < 4)).mean((1, 2, 3)) for l in range(16)], 1)
    cont = X[..., (wl > 690) & (wl < 710)].mean(-1)
    ti = band(wl, X, 497, 502); al = band(wl, X, 393, 398); ar = band(wl, X, 760, 767)
    gain = np.array([m["gain"] * (1 - m["lens_fouling"] * m["z_frac"]) for m in M[:n]])[:, None] * np.ones((1, 16))
    ee = e[:n]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    sc = ax[0].scatter(ee.ravel(), cont.ravel(), c=gain.ravel(), s=3, cmap="viridis")
    ax[0].set(xscale="log", yscale="log", xlabel="latent E/E_opt (layer)", ylabel="continuum @700 nm",
              title="X continuum vs latent\n(colour: gain x window transmission)")
    plt.colorbar(sc, ax=ax[0])
    ax[1].scatter(ee.ravel(), (al / np.maximum(ti, 1)).ravel(), s=3, c="#e76f51", alpha=0.4)
    ax[1].set(xscale="log", yscale="log", ylim=(0.05, 20), xlabel="latent E/E_opt", ylabel="Al/Ti line ratio",
              title="X line ratio (gain-invariant)")
    ax[2].scatter(ee.ravel(), np.maximum(layer_void.ravel() * 100, 1e-3), s=3, c="#264653", alpha=0.4)
    ax[2].set(xscale="log", yscale="log", xlabel="latent E/E_opt", ylabel="CT void fraction in layer (%)",
              title="U per-layer porosity vs latent")
    ax[3].scatter(np.maximum(ar.ravel(), 1), np.maximum(layer_void.ravel() * 100, 1e-3), s=3, alpha=0.4,
                  c=np.log(ee.ravel()), cmap="coolwarm")
    ax[3].set(xscale="log", yscale="log", xlabel="Ar I 763 nm peak (X)", ylabel="CT void fraction in layer (%)",
              title="X <-> U at layer level (colour: log E)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_shared_latent.png"), dpi=120); plt.close(fig)

    # ---------- 5. dataset statistics
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    splits = ["train", "val", "test"]
    sev = ["none", "minor", "moderate", "severe"]
    bottom = np.zeros(3)
    for s, c in zip(sev, ["#9aa0a6", "#8ecae6", "#f4a261", "#e63946"]):
        h = np.array([sum(m["split"] == sp and m["attributes"]["severity"] == s for m in M) for sp in splits])
        ax[0].bar(splits, h, bottom=bottom, color=c, label=s); bottom += h
    ax[0].legend(fontsize=8); ax[0].set_title("severity by split")
    cnt = collections.Counter(dom)
    ax[1].bar(list(cnt), list(cnt.values()), color=[DOM_COL[k] for k in cnt]); ax[1].set_title("dominant defect type")
    ax[1].tick_params(axis="x", labelsize=8)
    lens = collections.defaultdict(list)
    for t in T:
        lens[t["annotator"]].append(len(t["text"].split()))
    ax[2].hist([lens[k] for k in sorted(lens)], bins=30, label=[f"annotator {k}" for k in sorted(lens)], stacked=True)
    ax[2].legend(fontsize=8); ax[2].set(title="text length (words)", xlabel="words")
    ann = [m for m in M if m["annotated"]]
    c2 = collections.Counter(m["attributes"]["severity"] for m in ann)
    ax[3].bar(sev, [c2[s] for s in sev], color=["#9aa0a6", "#8ecae6", "#f4a261", "#e63946"])
    ax[3].set_title(f"annotated subset (n={len(ann)}) severity")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_stats.png"), dpi=120); plt.close(fig)

    # ---------- text samples
    with open(os.path.join(out, "sample_annotations.md"), "w") as fo:
        rng = np.random.default_rng(0)
        for sevk in sev:
            idx = [i for i, m in enumerate(M) if m["attributes"]["severity"] == sevk]
            fo.write(f"\n## {sevk} ({len(idx)} regions)\n\n")
            for i in rng.choice(idx, min(4, len(idx)), replace=False):
                fo.write(f"- `{M[i]['sample_id']}` [{T[i]['annotator']}] {T[i]['text']}\n")
    print("figures written to", out)


if __name__ == "__main__":
    main()
