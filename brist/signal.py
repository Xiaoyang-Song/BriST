"""Signal pathway: encoder f_X (spectra -> H_X) and projector g_X (H_X -> LLM tokens Z_X)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalEncoder(nn.Module):
    """Per-layer 1D CNN over wavelength, then a transformer across the 16 build layers.
    Input (B, 16, 1024) standardized log-spectra -> H_X (B, 16, d)."""

    def __init__(self, n_layers=16, d=256, tf_layers=2, dropout=0.1):
        super().__init__()
        c = [1, 32, 64, 96, 128]
        convs = []
        for i in range(4):
            convs += [nn.Conv1d(c[i], c[i + 1], 7, stride=2, padding=3), nn.GroupNorm(8, c[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*convs, nn.AdaptiveAvgPool1d(16))
        self.proj = nn.Linear(128 * 16, d)
        self.pos = nn.Parameter(torch.zeros(n_layers, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, tf_layers)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, x):
        B, L, C = x.shape
        h = self.cnn(x.reshape(B * L, 1, C)).flatten(1)
        h = self.proj(h).reshape(B, L, -1) + self.pos
        return self.norm(self.tf(h))

    @staticmethod
    def pool(H):
        return H.mean(1)


class SignalProjector(nn.Module):
    """Learned-query resampler: K queries cross-attend to H_X, then map to LLM width.
    Optionally predicts a per-token log-variance (for selective / uncertainty-aware alignment)."""

    def __init__(self, d=256, K=64, d_llm=960, n_blocks=2, hetero=False):
        super().__init__()
        self.q = nn.Parameter(torch.randn(K, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, n_blocks)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_llm))
        self.hetero = hetero
        if hetero:
            self.logvar = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, H):
        q = self.q.unsqueeze(0).expand(H.shape[0], -1, -1)
        h = self.dec(q, H)
        Z = self.out(h)
        if self.hetero:
            return Z, self.logvar(h).squeeze(-1)
        return Z


# ------------------------------------------------------------------ losses
def dcca_loss(H1, H2, r=10, eps=1e-3):
    """Negative sum of the top-r canonical correlations between two batches of features."""
    n = H1.shape[0]
    H1 = H1 - H1.mean(0); H2 = H2 - H2.mean(0)
    S11 = H1.T @ H1 / (n - 1) + eps * torch.eye(H1.shape[1], device=H1.device)
    S22 = H2.T @ H2 / (n - 1) + eps * torch.eye(H2.shape[1], device=H1.device)
    S12 = H1.T @ H2 / (n - 1)

    def inv_sqrt(S):
        e, V = torch.linalg.eigh(S)
        e = e.clamp_min(eps)
        return V @ torch.diag(e ** -0.5) @ V.T
    T = inv_sqrt(S11) @ S12 @ inv_sqrt(S22)
    s = torch.linalg.svdvals(T)
    return -s[:r].sum()


def token_mse(Zx, Zu, scale):
    """Eq. (7), normalised by the per-dimension variance of the teacher tokens."""
    return (((Zx - Zu) ** 2) / scale).mean()


def selective_token_loss(Zx, logvar, Zu, scale, rel):
    """Relevance-weighted heteroscedastic token alignment.
    rel (K,) : teacher semantic relevance of each token (mean 1);
    logvar (B,K): student's predicted unrecoverability of each token. Tokens that are
    relevant AND recoverable get the strongest pull; unrecoverable ones are down-weighted
    and their uncertainty is represented rather than forced into deterministic agreement."""
    err = (((Zx - Zu) ** 2) / scale).mean(-1)            # (B,K)
    nll = 0.5 * (err * torch.exp(-logvar) + logvar)
    return (nll * rel[None]).mean()


def kd_topk_loss(logits, mask, topk_idx, topk_logp, T=1.0):
    """Token-level KL(q_T || q_S) along teacher-generated prefixes, restricted to teacher top-k."""
    logq_s = F.log_softmax(logits / T, -1)
    ls = torch.gather(logq_s, -1, topk_idx)                # (B,L,k)
    pt = torch.softmax(topk_logp / T, -1)                  # renormalised teacher top-k
    kl = (pt * (torch.log(pt + 1e-9) - ls)).sum(-1)
    return (kl * mask).sum() / mask.sum()


class VectorEncoder(nn.Module):
    """f_X for low-dimensional signal vectors (e.g. the 16-d simulated spectra of the DCCA study):
    MLP -> T latent tokens -> small transformer. Output (B, T, d) like SignalEncoder."""

    def __init__(self, n_in, d=256, T=8, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, T * d))
        self.pos = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 1)
        self.norm = nn.LayerNorm(d)
        self.d, self.T = d, T

    def forward(self, x):
        h = self.mlp(x).reshape(x.shape[0], self.T, self.d) + self.pos
        return self.norm(self.tf(h))


def make_encoder(X):
    return VectorEncoder(X.shape[1]) if X.dim() == 2 else SignalEncoder()
