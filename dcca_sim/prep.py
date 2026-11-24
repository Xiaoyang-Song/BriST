"""Model-input cache for the circle/crack data: standardized 16-d spectra and the 32x32 CT images."""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dcca_sim")
ap.add_argument("--sigma", default="1.2")
ap.add_argument("--cache", default="runs_dcca/cache")
args = ap.parse_args()
os.makedirs(args.cache, exist_ok=True)
X = np.load(os.path.join(args.data, f"spectra_sigma{args.sigma}.npy")).astype(np.float32)
tr = np.arange(4000)
X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
np.save(os.path.join(args.cache, "X.npy"), X.astype(np.float32))
np.save(os.path.join(args.cache, "img.npy"), np.load(os.path.join(args.data, "images.npy")).astype(np.uint8))
print("cached", X.shape)
