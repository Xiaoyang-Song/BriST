"""Claim-level selective semantic transfer (post-hoc, no retraining).

For each attribute a (e.g. location, orientation, severity):
  1. pseudo-labels from the CT teacher's descriptions of all N training regions (bridge supervision);
  2. a probe p(a | f_X(x)) on the student's own signal features;
  3. transferability of a  = probe's val accuracy above the majority rate (normalised);
  4. at deployment, a generated claim about a is kept only if the probe confidence for the stated value
     exceeds a threshold chosen on validation to reach a target claim precision; otherwise the attribute is
     reported as "not resolvable from the in situ signal".
Reports, per attribute: stated-rate and claim accuracy before/after selection (risk-coverage), on test.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--students", nargs="+", default=["hier_kd", "selective_kd", "hier_vlm", "direct"])
ap.add_argument("--target-precision", type=float, default=0.8)
args = ap.parse_args()

d = load_dataset(args.data)
if d.name == "lpbf":
    from brist import textparse as tp
    ATTRS, claims, gtc = tp.CLAIM_ATTRS, tp.claims, tp.gt_claims_cat
else:
    from dcca_sim import textparse as tp
    ATTRS, claims, gtc = tp.CLAIM_ATTRS, tp.claims, tp.gt_claims
tr, va, te = d.idx("train"), d.idx("val"), d.idx("test")
teacher = json.load(open(os.path.join(args.root, "teacher/gen_A.json")))
T_cl = {i: claims(teacher[i]) for i in tr}
G = {i: gtc(d.meta[i]["attributes"]) for i in np.concatenate([va, te])}

out = {}
for name in args.students:
    sdir = os.path.join(args.root, "student", name)
    if not os.path.exists(os.path.join(sdir, "feats.npy")):
        continue
    F = np.load(os.path.join(sdir, "feats.npy"))
    gen = json.load(open(os.path.join(sdir, "gen_A.json")))
    S_cl = {i: claims(gen[i]) for i in np.concatenate([va, te])}
    sc = StandardScaler().fit(F[tr])
    Fz = sc.transform(F)
    res = {}
    for a in ATTRS:
        lab = [(i, T_cl[i][a]) for i in tr if T_cl[i][a] is not None]
        ys = [str(v) for _, v in lab]
        if len(set(ys)) < 2:
            continue
        probe = LogisticRegression(C=0.5, max_iter=3000).fit(Fz[[i for i, _ in lab]], ys)
        cls = list(probe.classes_)
        P = probe.predict_proba(Fz)

        def conf_correct(ids):
            c, ok = [], []
            for i in ids:
                v, g = S_cl[i][a], G[i][a]
                if v is None or g is None:
                    continue
                v = str(v)
                c.append(P[i, cls.index(v)] if v in cls else 0.0)
                ok.append(float(v == str(g)))
            return np.array(c), np.array(ok)
        # transferability: probe accuracy vs majority on val (against CT ground truth)
        gv = [(i, str(G[i][a])) for i in va if G[i][a] is not None]
        pacc = np.mean([cls[int(P[i].argmax())] == g for i, g in gv]) if gv else np.nan
        maj = max(np.mean([g == c for _, g in gv]) for c in set(g for _, g in gv)) if gv else np.nan
        transfer = (pacc - maj) / (1 - maj + 1e-9)
        cv, okv = conf_correct(va)
        thr = np.inf
        for t in np.sort(np.unique(np.concatenate([cv, [0.0]]))):
            keep = cv >= t
            if keep.sum() >= 5 and okv[keep].mean() >= args.target_precision:
                thr = t
                break
        ct, okt = conf_correct(te)
        n_te = sum(1 for i in te if G[i][a] is not None)
        keep = ct >= thr
        res[a] = dict(transferability=float(transfer), probe_acc=float(pacc), majority=float(maj), threshold=float(thr),
                      stated_before=float(len(ct) / max(n_te, 1)), acc_before=float(okt.mean()) if len(okt) else np.nan,
                      stated_after=float(keep.sum() / max(n_te, 1)),
                      acc_after=float(okt[keep].mean()) if keep.sum() else np.nan)
    out[name] = res
    print(f"\n[{name}]  target precision {args.target_precision}")
    print(f"{'attribute':12s} {'transfer':>8s} {'probe':>6s} {'major':>6s} | {'stated':>7s} {'acc':>6s} -> {'stated':>7s} {'acc':>6s}")
    for a, r in res.items():
        print(f"{a:12s} {r['transferability']:8.3f} {r['probe_acc']:6.3f} {r['majority']:6.3f} | "
              f"{r['stated_before']:7.3f} {r['acc_before']:6.3f} -> {r['stated_after']:7.3f} {r['acc_after']:6.3f}")
json.dump(out, open(os.path.join(args.root, "claim_selective.json"), "w"), indent=1)
