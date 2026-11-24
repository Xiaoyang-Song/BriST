"""Task 1: binary in situ process monitoring (defective = CT severity moderate/severe).

Score-based detectors (AUROC + thresholded metrics at the val-optimal threshold):
  sup_full   : spectra CNN (f_X + head) supervised with CT-derived labels for all N train regions
  sup_M      : same, labels only for the M annotated regions
  dcca_full / dcca_M : prior DCCA monitoring - DCCA-aligned f_X features + logistic regression
  <student>_feat_M   : student's aligned f_X features + logistic regression on M labels
Text alarms (severity stated in generated description; ordinal severity used as AUROC score):
  <student>_text, teacher_text (CT input, privileged reference)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402
from brist.signal import make_encoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--cache", default=None)
ap.add_argument("--students", nargs="*", default=[])
ap.add_argument("--out", default=None)
ap.add_argument("--root", default="runs")

ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
args.cache = args.cache or os.path.join(args.root, "cache")
torch.manual_seed(args.seed)

args.out = args.out or os.path.join(args.root, "monitor.json")
d = load_dataset(args.data)
y = d.defective
tr, va, te = d.idx("train"), d.idx("val"), d.idx("test")
trM = d.annotated_idx("train")
X = torch.from_numpy(np.load(os.path.join(args.cache, "X.npy"))).cuda()


def metrics(score, ids, thr):
    s, t = score[ids], y[ids]
    p = s >= thr
    tp, fp, fn = (p & (t == 1)).sum(), (p & (t == 0)).sum(), (~p & (t == 1)).sum()
    tn = (~p & (t == 0)).sum()
    return dict(auroc=float(roc_auc_score(t, s)), acc=float((p == t).mean()),
                bal_acc=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                f1=float(2 * tp / max(2 * tp + fp + fn, 1)), fpr=float(fp / max(fp + tn, 1)))


def best_thr(score):
    cands = np.unique(score[va])
    accs = [(0.5 * (((score[va] >= c) & (y[va] == 1)).sum() / max((y[va] == 1).sum(), 1)
                    + ((score[va] < c) & (y[va] == 0)).sum() / max((y[va] == 0).sum(), 1)), c) for c in cands]
    return max(accs)[1]


def sup_cnn(train_ids, steps=1500, bs=64):
    torch.manual_seed(args.seed)
    f = make_encoder(X).cuda(); head = nn.Linear(f.d, 1).cuda()
    ps = list(f.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(ps, lr=5e-4, weight_decay=1e-2)
    yt = torch.from_numpy(y).float().cuda()
    rng = np.random.default_rng(args.seed)
    best, best_sd = -1, None
    for s in range(1, steps + 1):
        b = rng.choice(train_ids, min(bs, len(train_ids)), replace=False)
        loss = nn.functional.binary_cross_entropy_with_logits(head(f(X[b]).mean(1)).squeeze(-1), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            f.eval()
            with torch.no_grad():
                sc = head(f(X[va]).mean(1)).squeeze(-1).cpu().numpy()
            f.train()
            auc = roc_auc_score(y[va], sc)
            if auc > best:
                best = auc
                best_sd = ({k: v.clone() for k, v in f.state_dict().items()}, {k: v.clone() for k, v in head.state_dict().items()})
    f.load_state_dict(best_sd[0]); head.load_state_dict(best_sd[1]); f.eval()
    with torch.no_grad():
        return torch.cat([head(f(X[s:s + 200]).mean(1)).squeeze(-1) for s in range(0, d.N, 200)]).cpu().numpy()


def logreg(feats, train_ids):
    sc = StandardScaler().fit(feats[train_ids])
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(feats[train_ids]), y[train_ids])
    return clf.decision_function(sc.transform(feats))


def enc_feats(path):
    f = make_encoder(X).cuda()
    f.load_state_dict(torch.load(path)["fx"]); f.eval()
    with torch.no_grad():
        return torch.cat([f(X[s:s + 200]).mean(1) for s in range(0, d.N, 200)]).cpu().numpy()


def text_score(texts):
    return np.array([d.alarm_score(t) if t else 0.0 for t in texts], float)


scores = {}
scores["sup_full"] = sup_cnn(tr)
scores["sup_M"] = sup_cnn(trM)
if os.path.exists(os.path.join(args.root, "align_feature/align.pt")):
    fd = enc_feats(os.path.join(args.root, "align_feature/align.pt"))
    scores["dcca_full"] = logreg(fd, tr)
    scores["dcca_M"] = logreg(fd, trM)
for name in args.students:
    p = os.path.join(args.root, "student", name)
    if os.path.exists(os.path.join(p, "feats.npy")):
        scores[f"{name}_feat_M"] = logreg(np.load(os.path.join(p, "feats.npy")), trM)
    if os.path.exists(os.path.join(p, "gen_A.json")):
        scores[f"{name}_text"] = text_score(json.load(open(os.path.join(p, "gen_A.json"))))
if os.path.exists(os.path.join(args.root, "teacher/gen_A.json")):
    scores["teacher_text(CT)"] = text_score(json.load(open(os.path.join(args.root, "teacher/gen_A.json"))))

res = {}
for k, s in scores.items():
    thr = d.alarm_threshold if k.endswith("_text") or k.endswith("(CT)") else best_thr(s)
    res[k] = {"val": metrics(s, va, thr), "test": metrics(s, te, thr)}
json.dump(res, open(args.out, "w"), indent=1)
print(f"{'method':32s} " + " ".join(f"{m:>8s}" for m in ["auroc", "acc", "bal_acc", "prec", "recall", "f1", "fpr"]))
for k, r in res.items():
    t = r["test"]
    print(f"{k:32s} " + " ".join(f"{t[m]:8.3f}" for m in ["auroc", "acc", "bal_acc", "precision", "recall", "f1", "fpr"]))
