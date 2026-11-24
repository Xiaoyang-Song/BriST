"""Task 2 summary: description fidelity on the test split for every student and the CT teacher.

Subsets: all test regions; truly defective; alarmed (method's own text raised the alarm).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--root", default="runs")
args = ap.parse_args()

d = load_dataset(args.data)
te = d.idx("test")
runs = {"teacher(CT)": os.path.join(args.root, "teacher/gen_A.json")}
sd = os.path.join(args.root, "student")
for n in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
    p = os.path.join(sd, n, "gen_A.json")
    if os.path.exists(p):
        runs[n] = p

out, cols = {}, None
for subset in ("all", "defective", "alarmed"):
    print(f"\n=== test / {subset} ===")
    for name, p in runs.items():
        g = json.load(open(p))
        if subset == "all":
            ids = te
        elif subset == "defective":
            ids = te[d.defective[te] == 1]
        else:
            ids = np.array([i for i in te if d.alarm_score(g[i]) >= d.alarm_threshold])
        if len(ids) == 0:
            continue
        agg, _ = d.score([g[i] for i in ids], [d.meta[i]["attributes"] for i in ids])
        agg["fidelity"] = d.fidelity(agg)
        if cols is None:
            cols = ["n", "fidelity"] + [k for k in agg if k not in ("n", "fidelity", "parse_rate")]
            print(f"{'method':24s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        out.setdefault(subset, {})[name] = agg
        print(f"{name:24s} " + " ".join(f"{agg[c]:9.3f}" for c in cols))
# specificity: do descriptions change with the input, and track the key physical quantity?
from collections import Counter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
de = te[d.defective[te] == 1]
print(f"\n=== specificity on truly defective test regions (n={len(de)}) ===")
print(f"{'method':24s} {'distinct':>9s} {'top1_share':>11s} {'spearman_' + d.quant_name:>26s}")
for name, p in runs.items():
    g = json.load(open(p))
    T = [g[i] for i in de]
    pairs = [d.quant(g[i], d.meta[i]["attributes"]) for i in de]
    pairs = [(a, b) for a, b in pairs if a is not None]
    rho = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).correlation if len(pairs) > 2 else float("nan")
    spec = dict(distinct=len(set(T)) / len(T), top1_share=Counter(T).most_common(1)[0][1] / len(T), spearman=float(rho))
    out.setdefault("specificity", {})[name] = spec
    print(f"{name:24s} {spec['distinct']:9.3f} {spec['top1_share']:11.3f} {spec['spearman']:26.3f}")
json.dump(out, open(os.path.join(args.root, "summary_task2.json"), "w"), indent=1)

rng = np.random.default_rng(0)
ex = rng.choice(te[d.defective[te] == 1], 4, replace=False).tolist() + rng.choice(te[d.defective[te] == 0], 2, replace=False).tolist()
with open(os.path.join(args.root, "examples.md"), "w") as f:
    for i in ex:
        a = d.meta[i]["attributes"]
        f.write(f"\n### test region {i}  (GT severity {a['severity']})\n")
        f.write(f"- **expert reference** [{d.text_all[i]['annotator']}]: {d.text_all[i]['text']}\n")
        for name, p in runs.items():
            f.write(f"- **{name}**: {json.load(open(p))[i]}\n")
print(f"\nwrote {args.root}/summary_task2.json, {args.root}/examples.md")
