"""Circle/crack simulation from the DCCA monitoring paper (reference/Source Code), unchanged in
substance, with ground-truth geometry recorded so the images can be annotated.

Copied (not imported, to keep reference/ untouched and dependency-free) from
reference/Source Code/models/simulation.py:
  generate_circle, generate_cracks, SimEncoder, SimDecoder, SimAutoEncoder, generate_2fc_waves
and from reference/Source Code/notebook/simulation-dgp.ipynb + jobs/simulation/gen_data.py:
  label sampling, image loop, AE training recipe, BETAS, V = U*BETAS + noise, 2fc spectra + noise.
The only change is that the drawing functions also return the parameters they drew.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 32
BETAS = np.array([0.3, 1, 2, 0.4, -0.5, -0.7])


# ------------------------------------------------------------------ images (reference generators + params)
def generate_circle(img_size, num_circles):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    params = []
    for _ in range(num_circles):
        radius = np.random.randint(img_size // 15, img_size // 5)
        center = (np.random.randint(radius, img_size - radius), np.random.randint(radius, img_size - radius))
        cv2.circle(img, center, radius, 1, -1)
        params.append(dict(center=[int(center[0]), int(center[1])], radius=int(radius)))
    return img, params


def generate_cracks(img_size, num_cracks):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    params = []
    for _ in range(num_cracks):
        start = (np.random.randint(img_size), np.random.randint(img_size))
        end = (np.random.randint(img_size), np.random.randint(img_size))
        thickness = np.random.randint(1, 2)
        cv2.line(img, start, end, 1, thickness)
        params.append(dict(start=[int(start[0]), int(start[1])], end=[int(end[0]), int(end[1])],
                           thickness=int(thickness)))
    return img, params


def generate_images(n, seed=2025, img_size=IMG_SIZE):
    """notebook/simulation-dgp.ipynb: Y ~ Bern(0.5); normal -> 1-2 circles, abnormal -> 1-2 cracks."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    Y = np.random.binomial(1, 0.5, n)
    data, params = [], []
    for i in range(n):
        if not Y[i]:
            num_circles = np.random.randint(1, 3)
            img, p = generate_circle(img_size, num_circles)
            params.append(dict(kind="circle", items=p))
        else:
            num = np.random.randint(1, 3)
            num_cracks = np.random.randint(1, 2)   # drawn but unused in the reference, kept for parity
            img, p = generate_cracks(img_size, num)
            params.append(dict(kind="crack", items=p))
        data.append(img)
    return np.array(data, dtype=np.float32), Y, params


# ------------------------------------------------------------------ DGP autoencoder (reference SimAutoEncoder)
class SimEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, hidden)

    def forward(self, x):
        x, indices1 = self.pool1(self.conv1(x))
        x, indices2 = self.pool2(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, indices1, indices2


class SimDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 512)
        self.fc2 = nn.Linear(512, 64 * 5 * 5)
        self.unpool1 = nn.MaxUnpool2d((2, 2))
        self.conv1 = nn.ConvTranspose2d(64, 32, 5)
        self.unpool2 = nn.MaxUnpool2d((2, 2))
        self.conv2 = nn.ConvTranspose2d(32, 1, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, indices1, indices2):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = out.reshape((-1, 64, 5, 5))
        out = self.unpool1(out, indices2)
        out = F.relu(self.conv1(out))
        out = self.unpool2(out, indices1)
        out = self.conv2(out)
        return self.sigmoid(out)


class SimAutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = SimEncoder(hidden_dim)
        self.decoder = SimDecoder(hidden_dim)

    def forward(self, x):
        output, indc1, indc2 = self.encoder(x)
        return self.decoder(output, indc1, indc2)


def train_dgp_autoencoder(images, h=6, epochs=500, bs=32, lr=1e-3, device="cuda", seed=2023, log=print):
    """reference run_sim_imgae: 80/20 split, Adam(lr, betas=(0.5,0.999)), StepLR(250, 0.1), MSE, 500 epochs."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(images).unsqueeze(1)
    perm = torch.randperm(len(x), generator=g)
    ntr = int(0.8 * len(x))
    tri, val = x[perm[:ntr]].to(device), x[perm[ntr:]].to(device)
    model = SimAutoEncoder(h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=250, gamma=0.1)
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(tri), generator=g)
        for s in range(0, len(tri), bs):
            b = tri[p[s:s + bs]]
            loss = F.mse_loss(model(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                vl = F.mse_loss(model(val), val).item()
            log(f"  AE epoch {ep + 1}/{epochs} train {loss.item():.4f} val {vl:.4f}")
    return model


# ------------------------------------------------------------------ spectra (jobs/simulation/gen_data.py, sp_type=2fc)
def generate_2fc_waves(V):
    M = torch.normal(1, 1, size=(6, 32), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    spectra = F.relu(spectra)
    M2 = torch.normal(0, 1, size=(32, 16), dtype=torch.float32)
    spectra = torch.log(torch.abs(1 / (1 - torch.matmul(spectra, M2))))
    return spectra, M, M2


def make_spectra(U, sigma, seed=2024):
    """gen_data.py: V = U*BETAS + N(0, 0.1^2); 2fc map with fixed (M, M2); + N(0, sigma^2) noise (as in test.sh)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(U)
    NOISE = np.random.normal(0, 0.1 ** 2, (n, 6))
    V = torch.tensor(U * BETAS + NOISE, dtype=torch.float32)
    _, M, M2 = generate_2fc_waves(V)
    sp = torch.matmul(V, M)
    sp = F.relu(sp)
    sp = torch.log(torch.abs(1 / (1 - torch.matmul(sp, M2)))) + np.random.normal(0, sigma ** 2, (n, 16))
    return sp.numpy().astype(np.float32), V.numpy(), M.numpy(), M2.numpy()
