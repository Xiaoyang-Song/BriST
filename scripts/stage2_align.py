"""Stage II: process-quality representation alignment on all N training (X, U) pairs.

--mode feature   : DCCA between pooled H_X and pooled vision features H_U   (Eq. 6)
--mode token     : MSE between Z_X and teacher tokens Z_U                   (Eq. 7)
--mode hier      : feature + token                                          (Eq. 8)
--mode selective : feature + relevance-weighted heteroscedastic token loss (Sec. 6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402
from brist.signal import (SignalProjector, dcca_loss, make_encoder, selective_token_loss,  # noqa: E402
                          token_mse)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["feature", "token", "hier", "selective"])
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--cache", default=None)
ap.add_argument("--teacher", default=None)
ap.add_argument("--root", default="runs")

ap.add_argument("--out", default=None)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lam-f", type=float, default=0.1)
ap.add_argument("--lam-s", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
args.cache = args.cache or os.path.join(args.root, "cache")
args.teacher = args.teacher or os.path.join(args.root, "teacher")
out = args.out or os.path.join(args.root, f"align_{args.mode}")
os.makedirs(out, exist_ok=True)
torch.manual_seed(args.seed)
dev = "cuda"

d = load_dataset(args.data)
X = torch.from_numpy(np.load(os.path.join(args.cache, "X.npy"))).to(dev)
hu = torch.from_numpy(np.load(os.path.join(args.teacher, "hu_pooled.npy"))).to(dev)   # adapted encoder
hu = (hu - hu[d.idx("train")].mean(0)) / (hu[d.idx("train")].std(0) + 1e-6)
ZU = torch.from_numpy(np.load(os.path.join(args.teacher, "ZU.npy"))).to(dev).float()
rel = torch.from_numpy(np.load(os.path.join(args.teacher, "relevance.npy"))).to(dev).float()
tr, va, te = d.idx("train"), d.idx("val"), d.idx("test")
scale = ZU[tr].var(0).mean(0, keepdim=True) + 1e-4       # (1, d) per-dim variance

use_f = args.mode in ("feature", "hier", "selective")
use_t = args.mode in ("token", "hier", "selective")
fx = make_encoder(X).to(dev)
gx = SignalProjector(hetero=args.mode == "selective").to(dev) if use_t else None
cca_dim = 32
hx_head = nn.Linear(fx.d, cca_dim).to(dev)
hu_head = nn.Sequential(nn.Linear(hu.shape[1], 256), nn.GELU(), nn.Linear(256, cca_dim)).to(dev)
params = list(fx.parameters()) + (list(gx.parameters()) if gx else []) + \
    (list(hx_head.parameters()) + list(hu_head.parameters()) if use_f else [])
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
steps = args.epochs * int(np.ceil(len(tr) / args.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)


def losses(ids, train=True):
    H = fx(X[ids])
    L = {}
    if use_f:
        L["feature"] = dcca_loss(hx_head(H.mean(1)), hu_head(hu[ids]), r=10)
    if use_t:
        if args.mode == "selective":
            Z, lv = gx(H)
            L["token"] = selective_token_loss(Z, lv, ZU[ids], scale, rel)
            L["token_mse"] = token_mse(Z, ZU[ids], scale).detach()
        else:
            Z = gx(H)
            L["token"] = token_mse(Z, ZU[ids], scale)
    tot = args.lam_f * L.get("feature", 0) + args.lam_s * L.get("token", 0)
    return tot, L


@torch.no_grad()
def evaluate(ids):
    fx.eval(); gx and gx.eval()
    _, L = losses(ids, False)
    r = {k: float(v) for k, v in L.items()}
    if use_t:
        H = fx(X[ids])
        Z = gx(H)[0] if args.mode == "selective" else gx(H)
        err = ((Z - ZU[ids]) ** 2).mean((0, 2))
        var = ((ZU[ids] - ZU[ids].mean(0)) ** 2).mean((0, 2))
        r["token_r2_mean"] = float((1 - err / var).mean())
        r["token_r2_per_token"] = (1 - err / var).cpu().numpy().round(3).tolist()
    fx.train(); gx and gx.train()
    return r


t0 = time.time()
rng = np.random.default_rng(args.seed)
step = 0
# checkpoint selection on the validation objective (token MSE if token alignment is used, else DCCA)
best_val, best_sd, best_ep = float("inf"), None, -1


def snapshot():
    return {"fx": {k: v.detach().clone() for k, v in fx.state_dict().items()},
            "gx": {k: v.detach().clone() for k, v in gx.state_dict().items()} if gx else None}
for ep in range(args.epochs):
    perm = rng.permutation(tr)
    for s in range(0, len(perm), args.bs):
        ids = perm[s:s + args.bs]
        if len(ids) < 32:
            continue
        tot, L = losses(ids)
        opt.zero_grad(set_to_none=True)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
    if (ep + 1) % 5 == 0:
        v = evaluate(va)
        crit = v["token_mse"] if "token_mse" in v else v.get("token", v.get("feature"))
        if crit < best_val:
            best_val, best_sd, best_ep = crit, snapshot(), ep + 1
    if (ep + 1) % 25 == 0:
        print(f"ep {ep + 1} " + " ".join(f"{k} {v[k]:.4f}" for k in v if not k.endswith("per_token"))
              + f" {time.time() - t0:.0f}s", flush=True)

fx.load_state_dict(best_sd["fx"])
if gx:
    gx.load_state_dict(best_sd["gx"])
print(f"selected epoch {best_ep} (val criterion {best_val:.4f})")
res = {"val": evaluate(va), "test": evaluate(te), "selected_epoch": best_ep}
torch.save({"fx": fx.state_dict(), "gx": gx.state_dict() if gx else None, "mode": args.mode,
            "hetero": args.mode == "selective"}, os.path.join(out, "align.pt"))
json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
print("test", {k: v for k, v in res["test"].items() if not k.endswith("per_token")}, "epoch", best_ep)
