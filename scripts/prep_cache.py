"""Cache model inputs: standardized log-spectra and frozen vision-encoder features of CT renderings."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.data import SynthData, ct_to_image, load_spectra, standardize  # noqa: E402
from brist.vlm import VLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic_v1")
ap.add_argument("--cache", default="runs/cache")
args = ap.parse_args()
os.makedirs(args.cache, exist_ok=True)

d = SynthData(args.data)
X = standardize(load_spectra(d), d.idx("train"))
np.save(os.path.join(args.cache, "X.npy"), X)

vlm = VLM(train_connector=False)
HU = np.lib.format.open_memmap(os.path.join(args.cache, "HU.npy"), mode="w+", dtype=np.float16, shape=(d.N, 1024, 768))
bs = 32
with d.h5() as f:
    for s in range(0, d.N, bs):
        ims = [ct_to_image(f["ct"][i]) for i in range(s, min(s + bs, d.N))]
        HU[s:s + len(ims)] = vlm.vision_features(ims).cpu().numpy()
        if s % 320 == 0:
            print(s, flush=True)
HU.flush()
np.save(os.path.join(args.cache, "hu_pooled.npy"), np.asarray(HU).astype(np.float32).mean(1))
print("done", X.shape, HU.shape)
