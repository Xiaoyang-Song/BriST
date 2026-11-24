"""Cache CT renderings (N,512,512,3) uint8 for vision-encoder adaptation."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brist.data import SynthData, ct_to_image  # noqa: E402
d = SynthData()
IM = np.lib.format.open_memmap("runs/cache/img.npy", mode="w+", dtype=np.uint8, shape=(d.N, 512, 512, 3))
with d.h5() as f:
    for i in range(d.N):
        IM[i] = np.asarray(ct_to_image(f["ct"][i]))
IM.flush(); print("done", IM.shape)
