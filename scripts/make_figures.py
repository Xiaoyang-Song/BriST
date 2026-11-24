"""Result figures for one dataset root.

python scripts/make_figures.py --data data/synthetic_v1 --root runs      --out figures/lpbf
python scripts/make_figures.py --data data/dcca_sim     --root runs_dcca --out figures/circlecrack
"""
import argparse
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# fixed entity -> colour (categorical slots in fixed order; CT reference in neutral ink)
COL = {"teacher(CT)": "#52514e", "direct": "#eb6834", "token_zeroshot": "#e87ba4", "feature_vlm": "#eda100",
       "hier_vlm": "#1baf7a", "hier_kd": "#2a78d6", "selective_kd": "#6250d6"}
LABEL = {"teacher(CT)": "CT teacher (privileged)", "direct": "Direct X→text (no CT)", "token_zeroshot": "Token align, zero-shot",
         "feature_vlm": "Feature (DCCA) align + VLM", "hier_vlm": "Hierarchical align + VLM",
         "hier_kd": "Hierarchical + KD", "selective_kd": "Selective + KD"}
ORDER = ["direct", "feature_vlm", "token_zeroshot", "hier_vlm", "hier_kd", "selective_kd", "teacher(CT)"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

d = load_dataset(args.data)
lp = d.name == "lpbf"
te = d.idx("test")
summ = json.load(open(os.path.join(args.root, "summary_task2.json")))
mon = json.load(open(os.path.join(args.root, "monitor.json")))
gens = {"teacher(CT)": json.load(open(os.path.join(args.root, "teacher/gen_A.json")))}
for m in ORDER[:-1]:
    p = os.path.join(args.root, "student", m, "gen_A.json")
    if os.path.exists(p):
        gens[m] = json.load(open(p))
methods = [m for m in ORDER if m in gens]
if lp:
    from brist import textparse as tp
    claims, gtc = tp.claims, tp.gt_claims_cat
else:
    from dcca_sim import textparse as tp
    claims, gtc = tp.claims, tp.gt_claims
X = np.load(os.path.join(args.root, "cache/X.npy"), mmap_mode="r")


def tidy(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ 1. qualitative examples
def claim_line(text, att):
    c, g = claims(text), gtc(att)
    parts = []
    for k, v in c.items():
        if v is None or g.get(k) is None:
            continue
        parts.append(("✓ " if str(v) == str(g[k]) else "✗ ") + k)
    return "   ".join(parts)


rng = np.random.default_rng(3)
de = te[d.defective[te] == 1]
nd = te[d.defective[te] == 0]
ex = list(rng.choice(de, 3, replace=False)) + list(rng.choice(nd, 1, replace=False))
show = ["teacher(CT)", "direct", "hier_kd"]
if lp:
    import h5py
    from brist.data import ct_to_image
    h5 = h5py.File(os.path.join(args.data, "dataset.h5"), "r")
    wl = h5["wavelength_nm"][:]
else:
    imgs = np.load(os.path.join(args.data, "images.npy"))
fig = plt.figure(figsize=(16, 4.3 * len(ex)))
gs = fig.add_gridspec(len(ex), 3, width_ratios=[1.15, 1.0, 2.6], wspace=0.12, hspace=0.35)
for r, i in enumerate(ex):
    att = d.meta[i]["attributes"]
    ax = fig.add_subplot(gs[r, 0])
    if lp:
        raw = h5["spectra"][i]
        ax.imshow(np.log10(np.clip(raw, 1, None)), aspect="auto", origin="lower", cmap="magma",
                  extent=[wl[0], wl[-1], 0.5, 16.5])
        ax.set(xlabel="wavelength (nm)", ylabel="build layer")
        ax.set_title(f"In situ spectra X (log counts), region {i}", fontsize=9, color=INK, loc="left")
    else:
        ax.bar(np.arange(16), np.asarray(X[i]), color="#2a78d6", width=0.7)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set(xlabel="spectral channel", ylabel="standardized intensity")
        ax.set_title(f"In situ signal X (16-d), region {i}", fontsize=9, color=INK, loc="left")
        tidy(ax, "y")
    ax2 = fig.add_subplot(gs[r, 1])
    if lp:
        ax2.imshow(ct_to_image(h5["ct"][i]))
        ax2.set_title("Offline CT U (MinIP views; training only)", fontsize=9, color=INK, loc="left")
    else:
        ax2.imshow(imgs[i], cmap="gray_r", vmin=0, vmax=1.6)
        for k in (32 / 3, 64 / 3):
            ax2.axhline(k - 0.5, color=GRID, lw=0.6); ax2.axvline(k - 0.5, color=GRID, lw=0.6)
        ax2.set_title("Offline CT U (training only; 3×3 location grid)", fontsize=9, color=INK, loc="left")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(gs[r, 2]); ax3.axis("off")
    y = 1.0
    blocks = [("Expert reference (CT-derived ground truth)", d.text_all[i]["text"], INK, None)]
    for m in show:
        blocks.append((LABEL[m] + (" — sees CT" if m == "teacher(CT)" else " — sees only X"), gens[m][i], COL[m],
                       claim_line(gens[m][i], att)))
    for title, txt, c, cl in blocks:
        ax3.text(0, y, title, color=c, fontsize=8.5, fontweight="bold", va="top", transform=ax3.transAxes)
        y -= 0.075
        w = "\n".join(textwrap.wrap(txt, 125))
        ax3.text(0, y, w, color=INK, fontsize=8, va="top", transform=ax3.transAxes, family="DejaVu Sans")
        y -= 0.062 * (w.count("\n") + 1)
        if cl:
            ax3.text(0, y, cl, color=INK2, fontsize=7.5, va="top", transform=ax3.transAxes)
            y -= 0.07
        y -= 0.02
fig.subplots_adjust(top=0.965)
fig.suptitle("Spectra → description: examples from held-out test builds (✓/✗ = claim vs CT ground truth)", y=0.985,
             x=0.01, ha="left", fontsize=11, color=INK)
fig.savefig(os.path.join(args.out, "qualitative_examples.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 2. task 1: binary monitoring
rows = [("sup_full", "Spectra CNN, CT labels for all N", "#9a9893"),
        ("sup_M", "Spectra CNN, labels for M only", "#c3c2b7"),
        ("dcca_full", "DCCA features + LR, all N labels", "#9a9893"),
        ("direct_text", LABEL["direct"] + " (text alarm)", COL["direct"]),
        ("token_zeroshot_text", LABEL["token_zeroshot"] + " (text alarm)", COL["token_zeroshot"]),
        ("hier_vlm_text", LABEL["hier_vlm"] + " (text alarm)", COL["hier_vlm"]),
        ("hier_kd_text", LABEL["hier_kd"] + " (text alarm)", COL["hier_kd"]),
        ("selective_kd_text", LABEL["selective_kd"] + " (text alarm)", COL["selective_kd"]),
        ("teacher_text(CT)", LABEL["teacher(CT)"] + " (text alarm)", COL["teacher(CT)"])]
rows = [r for r in rows if r[0] in mon]
fig, axs = plt.subplots(1, 3, figsize=(15, 0.42 * len(rows) + 1.4), sharey=True)
for ax, (k, t) in zip(axs, [("bal_acc", "Balanced accuracy"), ("recall", "Recall (defects caught)"), ("fpr", "False-alarm rate")]):
    vals = [mon[r[0]]["test"][k] for r in rows]
    ax.barh(range(len(rows)), vals, color=[r[2] for r in rows], height=0.6)
    for j, v in enumerate(vals):
        ax.text(v + 0.01, j, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_title(t, loc="left", fontsize=10, color=INK)
    ax.set_xlim(0, 1.1 if k != "fpr" else max(vals) * 1.3 + 0.02)
    tidy(ax)
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[1] for r in rows]); axs[0].invert_yaxis()
fig.suptitle("Task 1 — binary in situ monitoring on held-out test builds (grey = spectra-only baselines)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "task1_monitoring.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 3. task 2: attribute fidelity on defective regions
S = summ["defective"]
attrs = ([("sev_acc", "severity"), ("dom_acc", "dominant type"), ("type_prec_major", "type precision"),
          ("z_hit", "layer localisation"), ("void_logerr", "void-fraction log10 error ↓")] if lp else
         [("type_acc", "defect type"), ("sev_acc", "severity"), ("count_acc", "count"), ("loc_acc", "location"),
          ("orient_acc", "orientation"), ("size_relerr", "size rel. error ↓")])
fig, axs = plt.subplots(1, len(attrs), figsize=(2.6 * len(attrs), 3.6), sharey=True)
ms = [m for m in methods if m in S]
for ax, (k, t) in zip(axs, attrs):
    vals = [S[m][k] for m in ms]
    ax.barh(range(len(ms)), vals, color=[COL[m] for m in ms], height=0.62)
    for j, v in enumerate(vals):
        ax.text(v, j, f" {v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_title(t, loc="left", fontsize=9.5, color=INK)
    ax.set_xlim(0, max(1.0, max(vals) * 1.25) if "↓" not in t else max(vals) * 1.35)
    tidy(ax)
axs[0].set_yticks(range(len(ms))); axs[0].set_yticklabels([LABEL[m] for m in ms]); axs[0].invert_yaxis()
fig.suptitle(f"Task 2 — description fidelity on truly defective test regions (n={S[ms[0]]['n']}); "
             "claims parsed from text and checked against CT", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "task2_attributes.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 4. specificity (mode collapse) + quantity tracking
sp = summ["specificity"]
ms = [m for m in methods if m in sp]
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
for ax, (k, t) in zip(axs[:2], [("distinct", "Distinct descriptions / region"),
                                ("top1_share", "Share of the single most frequent text ↓")]):
    vals = [sp[m][k] for m in ms]
    ax.barh(range(len(ms)), vals, color=[COL[m] for m in ms], height=0.62)
    for j, v in enumerate(vals):
        ax.text(v, j, f" {v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LABEL[m] for m in ms] if ax is axs[0] else [])
    ax.invert_yaxis(); ax.set_xlim(0, 1.15); ax.set_title(t, loc="left", fontsize=10, color=INK); tidy(ax)
ax = axs[2]
for m in [x for x in ("direct", "hier_kd", "teacher(CT)") if x in gens]:
    pairs = [d.quant(gens[m][i], d.meta[i]["attributes"]) for i in de]
    pairs = np.array([(a, b) for a, b in pairs if a is not None], float)
    ax.scatter(pairs[:, 1], pairs[:, 0], s=14, color=COL[m], alpha=0.55, edgecolor=SURF, lw=0.5,
               label=f"{LABEL[m]} (ρ={sp[m]['spearman']:.2f})")
lim = [np.nanmin(pairs), np.nanmax(pairs)]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
if lp:
    ax.set(xscale="log", yscale="log")
ax.set(xlabel=f"true {d.quant_name.replace('_', ' ')} (CT)", ylabel="stated in description")
ax.set_title("Does the stated quantity follow the input?", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=7.5, frameon=False); tidy(ax, "both")
fig.suptitle("Specificity — with few annotations the direct model collapses to a memorised report",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "specificity.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 5. annotation efficiency (if sweep present)
ratios_default = 0.10
pts = {}
for base in ("direct", "selective_kd"):
    for r in (0.02, 0.05, None, 0.25):
        name = base if r is None else f"{base}_r{r}"
        p = os.path.join(args.root, "student", name, "metrics.json")
        if os.path.exists(p):
            mt = json.load(open(p))
            pts.setdefault(base, []).append((r or ratios_default, mt["n_annot"], mt["final"]["test"]))
if pts:
    keys = ([("sev_acc", "severity accuracy"), ("dom_acc", "dominant-type accuracy"), ("alarm_acc", "alarm accuracy")]
            if lp else [("type_acc", "defect-type accuracy"), ("sev_acc", "severity accuracy"), ("count_acc", "count accuracy")])
    fig, axs = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 3.4), sharex=True)
    for ax, (k, t) in zip(axs, keys):
        for base, v in pts.items():
            v = sorted(v)
            ax.plot([x[0] * 100 for x in v], [x[2][k] for x in v], color=COL[base], lw=2, marker="o", ms=6,
                    label=LABEL[base])
        tref = summ["all"]["teacher(CT)"][k] if k in summ["all"]["teacher(CT)"] else None
        if tref is not None:
            ax.axhline(tref, color=COL["teacher(CT)"], lw=1, ls="--", label="CT teacher (10% labels)")
        ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 25]); ax.set_xticklabels(["2%", "5%", "10%", "25%"])
        ax.set_title(t, loc="left", fontsize=10, color=INK); ax.set_xlabel("expert-annotated fraction of training regions")
        tidy(ax, "both")
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Annotation efficiency on held-out test builds", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "annotation_efficiency.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------ 6. claim-level selective transfer (if present)
cp = os.path.join(args.root, "claim_selective.json")
if os.path.exists(cp):
    cs = json.load(open(cp))
    m = "hier_kd" if "hier_kd" in cs else list(cs)[0]
    r = cs[m]
    at = list(r)
    fig, axs = plt.subplots(1, 3, figsize=(15, 0.5 * len(at) + 1.6), sharey=True)
    axs[0].barh(range(len(at)), [r[a]["transferability"] for a in at], color="#2a78d6", height=0.6)
    axs[0].axvline(0, color=INK2, lw=0.8)
    axs[0].set_title("Transferability from X\n(probe acc. above majority, normalised)", loc="left", fontsize=9.5, color=INK)
    for ax, k, t in ((axs[1], "acc", "Claim accuracy when stated"), (axs[2], "stated", "Fraction of regions with the claim")):
        ax.barh(np.arange(len(at)) - 0.17, [r[a][f"{k}_before"] for a in at], height=0.32, color="#c3c2b7", label="all claims")
        ax.barh(np.arange(len(at)) + 0.17, [np.nan_to_num(r[a][f"{k}_after"]) for a in at], height=0.32, color="#2a78d6",
                label="after claim-level selection")
        ax.set_xlim(0, 1.05); ax.set_title(t, loc="left", fontsize=9.5, color=INK); tidy(ax)
    axs[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in axs:
        tidy(ax)
    axs[0].set_yticks(range(len(at))); axs[0].set_yticklabels(at); axs[0].invert_yaxis()
    fig.suptitle(f"Claim-level selective transfer ({LABEL[m]}): keep a claim only if the signal supports it",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "claim_selective.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
print("figures ->", args.out, sorted(os.listdir(args.out)))
