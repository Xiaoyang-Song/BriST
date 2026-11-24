"""Stage III: signal-to-language student (and baselines).

Variants are set by flags:
  direct            --decoder base                       (no CT, no alignment; M texts only)
  feature+VLM       --align runs/align_feature           (DCCA-pretrained f_X, teacher decoder, M texts)
  token zero-shot   --align runs/align_token --zero-shot (Z_X fed to frozen teacher decoder)
  hier+VLM          --align runs/align_hier
  hier+KD           --align runs/align_hier --kd 1
  selective+KD      --align runs/align_selective --kd 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.datasets import load_dataset  # noqa: E402
from brist.signal import (SignalProjector, kd_topk_loss, make_encoder, selective_token_loss,  # noqa: E402
                          token_mse)
from brist.vlm import VLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--cache", default=None)
ap.add_argument("--teacher", default=None)
ap.add_argument("--root", default="runs")

ap.add_argument("--align", default=None)
ap.add_argument("--decoder", default="teacher", choices=["teacher", "base"])
ap.add_argument("--zero-shot", action="store_true")
ap.add_argument("--kd", type=float, default=0.0)
ap.add_argument("--tok", type=float, default=0.0, help="keep token alignment as a regulariser")
ap.add_argument("--ratio", type=float, default=None, help="annotation ratio (default: shipped 10%% set)")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--eval-every", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
args.cache = args.cache or os.path.join(args.root, "cache")
args.teacher = args.teacher or os.path.join(args.root, "teacher")
out = os.path.join(args.root, "student", args.name)
os.makedirs(out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

d = load_dataset(args.data)
X = torch.from_numpy(np.load(os.path.join(args.cache, "X.npy"))).cuda()
vlm = VLM(train_connector=False)
if args.decoder == "teacher":
    vlm.load_trainable(torch.load(os.path.join(args.teacher, "teacher.pt")))
for p in vlm.connector.parameters():
    p.requires_grad = False

hetero = False
fx = make_encoder(X).cuda()
gx = None
if args.align:
    ck = torch.load(os.path.join(args.align, "align.pt"))
    fx.load_state_dict(ck["fx"])
    hetero = ck["hetero"]
    if ck["gx"] is not None:
        gx = SignalProjector(hetero=hetero).cuda()
        gx.load_state_dict(ck["gx"])
if gx is None:
    gx = SignalProjector().cuda()


def zx(ids):
    H = fx(X[ids])
    Z = gx(H)
    return (Z[0], Z[1], H) if hetero else (Z, None, H)


tr = d.annotated_idx("train", args.ratio, args.seed)
trall, va, te = d.idx("train"), d.idx("val"), d.idx("test")
print(f"[{args.name}] annotated train pairs: {len(tr)}  kd={args.kd} tok={args.tok}", flush=True)

if args.kd > 0 or args.tok > 0:
    ZU = torch.from_numpy(np.load(os.path.join(args.teacher, "ZU.npy"))).cuda().float()
    scale = ZU[trall].var(0).mean(0, keepdim=True) + 1e-4
    rel = torch.from_numpy(np.load(os.path.join(args.teacher, "relevance.npy"))).cuda().float()
if args.kd > 0:
    kd = torch.load(os.path.join(args.teacher, "kd_topk.pt"))
    kd_pos = {i: k for k, i in enumerate(kd["idx"])}


@torch.no_grad()
def gen(ids, bs=50):
    vlm.eval(); fx.eval(); gx.eval()
    res = []
    for s in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            res += vlm.generate(zx(ids[s:s + bs])[0], "A")
    vlm.train(); fx.train(); gx.train()
    return res


def fidelity(agg):
    return d.fidelity(agg)


def state():
    return {"lora": vlm.trainable_state(), "fx": {k: v.cpu().clone() for k, v in fx.state_dict().items()},
            "gx": {k: v.cpu().clone() for k, v in gx.state_dict().items()}}


hist = []
best, best_state = -1, None
t0 = time.time()
if not args.zero_shot:
    sig_lr = 1e-4 if args.align else 5e-4
    opt = torch.optim.AdamW([{"params": vlm.lora_params(), "lr": 1e-4},
                             {"params": list(fx.parameters()) + list(gx.parameters()), "lr": sig_lr}],
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, sig_lr], total_steps=args.steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    params = vlm.lora_params() + list(fx.parameters()) + list(gx.parameters())
    for step in range(1, args.steps + 1):
        b = rng.choice(tr, min(args.bs, len(tr)), replace=False)
        with torch.autocast("cuda", dtype=torch.float16):
            Z, lv, _ = zx(b)
            loss = vlm.lm_loss(Z, [d.text_all[i]["annotator"] for i in b], [d.text_all[i]["text"] for i in b])
            logs = {"ce": loss.item()}
            if args.kd > 0 or args.tok > 0:
                bk = rng.choice(trall, args.bs, replace=False)
                Zk, lvk, _ = zx(bk)
                if args.kd > 0:
                    ans = [kd["ids"][kd_pos[i]] for i in bk]
                    logits, mask, _ = vlm.forward_logits(Zk, ["A"] * len(bk), ans)
                    Lm = logits.shape[1]
                    ti = torch.zeros(len(bk), Lm, kd["topk_idx"][0].shape[-1], dtype=torch.long, device="cuda")
                    tl = torch.zeros(len(bk), Lm, ti.shape[-1], device="cuda")
                    for k, i in enumerate(bk):
                        a, v = kd["topk_idx"][kd_pos[i]], kd["topk_logp"][kd_pos[i]]
                        ti[k, :len(a)] = a.cuda().long(); tl[k, :len(v)] = v.cuda().float()
                    lkd = kd_topk_loss(logits, mask.float(), ti, tl)
                    loss = loss + args.kd * lkd
                    logs["kd"] = lkd.item()
                if args.tok > 0:
                    lt = (selective_token_loss(Zk.float(), lvk.float(), ZU[bk], scale, rel) if hetero
                          else token_mse(Zk.float(), ZU[bk], scale))
                    loss = loss + args.tok * lt
                    logs["tok"] = lt.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 50 == 0:
            print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in logs.items()) + f" {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            agg, _ = d.score(gen(va), [d.meta[i]["attributes"] for i in va])
            f = fidelity(agg)
            hist.append(dict(step=step, val_fid=f, **agg))
            print(f"  [val] step {step} fid {f:.3f} sev {agg['sev_acc']:.3f} "
                  f"alarm_acc {agg['alarm_acc']:.3f}", flush=True)
            if f > best:
                best, best_state = f, state()
    vlm.load_trainable(best_state["lora"]); fx.load_state_dict(best_state["fx"]); gx.load_state_dict(best_state["gx"])
    torch.save(best_state, os.path.join(out, "student.pt"))

# ---- final evaluation
ev = np.concatenate([va, te])
texts = [""] * d.N
for i, t in zip(ev, gen(ev)):
    texts[i] = t
json.dump(texts, open(os.path.join(out, "gen_A.json"), "w"))
res = {}
for sp in ("val", "test"):
    ids = d.idx(sp)
    res[sp], _ = d.score([texts[i] for i in ids], [d.meta[i]["attributes"] for i in ids])
with torch.no_grad():
    fx.eval()
    feats = torch.cat([fx(X[s:s + 200]).mean(1) for s in range(0, d.N, 200)]).cpu().numpy()
np.save(os.path.join(out, "feats.npy"), feats)
json.dump(dict(args=vars(args), n_annot=len(tr), history=hist, final=res), open(os.path.join(out, "metrics.json"), "w"), indent=1)
print("test:", {k: round(v, 3) for k, v in res["test"].items()})
print(f"done {time.time() - t0:.0f}s")
