"""Stage I: domain-specific VLM adaptation on the M annotated CT-text pairs (QLoRA + projector).

Outputs (runs/teacher/):
  teacher.pt           trainable params (LoRA + connector)
  ZU.npy               teacher CT tokens Z_U for all N regions (frozen reference for Stage II)
  gen_A.json           teacher detailed descriptions for all N regions
  kd_topk.pt           teacher top-k log-probs along its own descriptions (train split) for KD
  relevance.npy        per-token semantic relevance of Z_U (gradient saliency)
  metrics.json
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
from brist.vlm import VLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--cache", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--root", default="runs")
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--eval-every", type=int, default=200)
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
args.out = args.out or os.path.join(args.root, "teacher")
args.cache = args.cache or os.path.join(args.root, "cache")
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)

d = load_dataset(args.data)
IM = np.load(os.path.join(args.cache, "img.npy"), mmap_mode="r")
vlm = VLM(vision_lora=True)
dev = vlm.device
tr = d.annotated_idx("train")
va = d.idx("val")
print(f"teacher: {len(tr)} annotated CT-text pairs")


def img(ids):
    ids = np.asarray(ids)
    order = np.argsort(ids)                      # sorted reads from the memmap
    x = np.asarray(IM[ids[order]])
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    x = torch.from_numpy(x[inv]).to(dev)
    if x.dim() == 3:                             # small binary CT image (circle/crack data) -> 512 RGB
        x = (200.0 - 160.0 * x.float())[:, None]
        x = torch.nn.functional.interpolate(x, size=512, mode="nearest").repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
    return x


def hu(ids):
    """vision tokens H_U from the (LoRA-adapted) vision encoder"""
    return vlm.encode_pixels(img(ids))


def gen_texts(ids, bs=50):
    out = []
    vlm.eval()
    for s in range(0, len(ids), bs):
        b = np.asarray(ids[s:s + bs])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out += vlm.generate(vlm.connect(hu(b)), "A")
    return out


lrs = [2e-4, 1e-4, 1e-4]
opt = torch.optim.AdamW([{"params": vlm.lora_params(), "lr": lrs[0]},
                         {"params": vlm.vision_lora_params(), "lr": lrs[1]},
                         {"params": vlm.connector_params(), "lr": lrs[2]}], weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=args.steps, pct_start=0.1)
print("trainable: dec-LoRA %d, vis-LoRA %d, connector %d" % tuple(
    sum(p.numel() for p in g["params"]) for g in opt.param_groups), flush=True)
scaler = torch.cuda.amp.GradScaler()
best, best_state, hist = -1, None, []
rng = np.random.default_rng(args.seed)
t0 = time.time()
for step in range(1, args.steps + 1):
    vlm.train()
    b = np.sort(rng.choice(tr, args.bs, replace=False))
    texts = [d.text_all[i]["text"] for i in b]
    styles = [d.text_all[i]["annotator"] for i in b]
    with torch.autocast("cuda", dtype=torch.float16):
        loss = vlm.lm_loss(vlm.connect(hu(b)), styles, texts)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(vlm.lora_params() + vlm.vision_lora_params() + vlm.connector_params(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if step % 25 == 0:
        print(f"step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        txt = gen_texts(va)
        agg, _ = d.score(txt, [d.meta[i]["attributes"] for i in va])
        fid = d.fidelity(agg)
        hist.append(dict(step=step, val_fid=fid, **agg))
        print(f"  [val] step {step} fid {fid:.3f} sev {agg['sev_acc']:.3f} alarm_acc {agg['alarm_acc']:.3f} "
              f"| {txt[0][:150]}", flush=True)
        if fid > best:
            best, best_state = fid, vlm.trainable_state()

vlm.load_trainable(best_state)
torch.save(best_state, os.path.join(args.out, "teacher.pt"))
vlm.eval()

# ---- Z_U for all regions
ZU = np.zeros((d.N, 64, vlm.d), np.float16)
HP = np.zeros((d.N, 768), np.float32)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for s in range(0, d.N, 64):
        H = hu(np.arange(s, min(s + 64, d.N)))
        HP[s:s + 64] = H.float().mean(1).cpu().numpy()
        ZU[s:s + 64] = vlm.connect(H).float().cpu().numpy()
np.save(os.path.join(args.out, "ZU.npy"), ZU)
np.save(os.path.join(args.out, "hu_pooled.npy"), HP)     # adapted H_U for feature-level alignment

# ---- teacher descriptions for all regions
ev_ids = np.where(np.isin(d.split, ["train", "val", "test"]))[0]
allg = [""] * d.N
for i, t in zip(ev_ids, gen_texts(ev_ids)):
    allg[i] = t
json.dump(allg, open(os.path.join(args.out, "gen_A.json"), "w"))
res = {}
for sp in ("val", "test"):
    ids = d.idx(sp)
    res[sp], _ = d.score([allg[i] for i in ids], [d.meta[i]["attributes"] for i in ids])
res["train_unannotated"], _ = d.score([allg[i] for i in d.idx("train") if not d.annotated[i]],
                                    [d.meta[i]["attributes"] for i in d.idx("train") if not d.annotated[i]])
print("teacher test:", {k: round(v, 3) for k, v in res["test"].items()})

# ---- KD targets: teacher top-k along its own detailed descriptions (train split)
trall = d.idx("train")
kd = {"idx": trall.tolist(), "ids": [], "topk_idx": [], "topk_logp": []}
with torch.no_grad():
    for s in range(0, len(trall), 32):
        b = trall[s:s + 32]
        ans = vlm.encode_answers([allg[i] for i in b])
        with torch.autocast("cuda", dtype=torch.float16):
            logits, mask, _ = vlm.forward_logits(vlm.connect(hu(b)), ["A"] * len(b), ans)
        lp = torch.log_softmax(logits.float(), -1)
        v, ix = lp.topk(args.topk, -1)
        for k, a in enumerate(ans):
            kd["ids"].append(a)
            kd["topk_idx"].append(ix[k, :len(a)].to(torch.int32).cpu())
            kd["topk_logp"].append(v[k, :len(a)].half().cpu())
torch.save(kd, os.path.join(args.out, "kd_topk.pt"))

# ---- semantic relevance of each Z_U token (gradient saliency of the expert-text likelihood)
rel = torch.zeros(64, device=dev)
for s in range(0, len(tr), 8):
    b = tr[s:s + 8]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        Z = vlm.connect(hu(b)).float()
    Z.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = vlm.lm_loss(Z, [d.text_all[i]["annotator"] for i in b], [d.text_all[i]["text"] for i in b])
    g, = torch.autograd.grad(loss, Z)
    rel += g.norm(dim=-1).sum(0)
rel = (rel / rel.mean()).cpu().numpy()
np.save(os.path.join(args.out, "relevance.npy"), rel)
json.dump(dict(history=hist, final=res, n_train=len(tr)), open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s")
