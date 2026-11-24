"""Generate the circle/crack simulation (reference DGP) with expert-style annotations.

python -m dcca_sim.generate --out data/dcca_sim --sigma 1.2
Outputs: images.npy (N,32,32), Y.npy, U.npy (AE latent), V.npy, spectra_sigma{s}.npy (N,16),
         ct_ae-dgp.pt, metadata.jsonl (split, annotated, attributes, text, annotator), config.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .annotate import describe, measure
from .dgp import generate_images, make_spectra, train_dgp_autoencoder

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/dcca_sim")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--sigma", type=float, nargs="+", default=[1.2])
ap.add_argument("--ae-epochs", type=int, default=500)
ap.add_argument("--device", default="cuda")
ap.add_argument("--annot-frac", type=float, default=0.10)
ap.add_argument("--annotator-probs", type=float, nargs=3, default=[0.45, 0.40, 0.15])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()

imgs, Y, params = generate_images(args.n, seed=2025)
np.save(os.path.join(args.out, "images.npy"), imgs.astype(np.uint8))
np.save(os.path.join(args.out, "Y.npy"), Y)
print(f"images {imgs.shape}, abnormal frac {Y.mean():.3f}", flush=True)

ae_path = os.path.join(args.out, "ct_ae-dgp.pt")
if os.path.exists(ae_path):
    from .dgp import SimAutoEncoder
    ae = SimAutoEncoder(6).to(args.device); ae.load_state_dict(torch.load(ae_path, map_location=args.device))
else:
    ae = train_dgp_autoencoder(imgs, epochs=args.ae_epochs, device=args.device)
    torch.save(ae.state_dict(), ae_path)
ae.eval()
with torch.no_grad():
    x = torch.tensor(imgs).unsqueeze(1).to(args.device)
    U = torch.cat([ae.encoder(x[s:s + 1000])[0] for s in range(0, len(x), 1000)]).cpu().numpy()
np.save(os.path.join(args.out, "U.npy"), U)
for s in args.sigma:
    sp, V, M, M2 = make_spectra(U, s)
    np.save(os.path.join(args.out, f"spectra_sigma{s}.npy"), sp)
np.save(os.path.join(args.out, "V.npy"), V)
print(f"AE + spectra done {time.time() - t0:.0f}s", flush=True)

# splits as in the reference: first half train (here 4000 train / 1000 val), second half test
split = np.array(["train"] * 4000 + ["val"] * 1000 + ["test"] * (args.n - 5000))
rng = np.random.default_rng(0)
pool = np.where(split != "test")[0]
ann = set(rng.choice(pool, int(round(args.annot_frac * len(pool))), replace=False).tolist())
probs = np.asarray(args.annotator_probs) / sum(args.annotator_probs)
with open(os.path.join(args.out, "metadata.jsonl"), "w") as f:
    for i in range(args.n):
        att = measure(imgs[i], params[i])
        style = "ABC"[rng.choice(3, p=probs)]
        f.write(json.dumps(dict(idx=i, split=split[i], annotated=i in ann, Y=int(Y[i]), params=params[i],
                                attributes=att, annotator=style, text=describe(att, rng, style))) + "\n")
json.dump(dict(args=vars(args), px_um=10, betas=[0.3, 1, 2, 0.4, -0.5, -0.7]),
          open(os.path.join(args.out, "config.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s; annotated {len(ann)} of {len(pool)} train+val")
