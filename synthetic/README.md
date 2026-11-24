# Synthetic BriST data (v1)

Simulated LPBF Ti-6Al-4V regions with paired in situ optical emission spectra (X),
offline X-ray CT (U), and expert-style quality descriptions (Y). X and U are both
rendered from one shared latent process state; Y is written only from what CT shows.

```
python -m synthetic.generate --out data/synthetic_v1      # ~10 min, 2000 regions, ~0.5 GB
python -m synthetic.review   --data data/synthetic_v1     # figures in data/synthetic_v1/review/
```

## Generative structure

```
build  (10)  powder moisture ─────────────┬─> gas pores (U)        ─┐
             chamber O2, gain, dark,      └─> H I 656 nm line (X)   │  X-only nuisance:
             lens fouling, wl shift ────────> X only                │  gain, fouling, O2,
part   (10/build)  laser power P, speed v  -> nominal VED (process-map DOE:  │  wl shift, stray light
             LoF / nominal / keyhole), scan rotation 67°/layer,              │
             CT scan settings ─────────────> U only (noise, PSF, cupping, rings)
region (20/part)  16 layers x 30 µm; heat accumulation with height,
             AR(1) energy drift, power-dip (12%) / overheating (8%) events
layer        e = VED_eff / 70 J/mm³   ── shared latent ──
             ├─ melt-pool T, area  ──> Planck continuum (X)
             ├─ vapor / plume      ──> Ti, V, Al, Ar lines, Al/Ti ratio, Stark width (X)
             ├─ instability        ──> frame fluctuation, spatter (X)
             ├─ s_kh(e), depth     ──> keyhole pores at melt-pool depth (U)
             └─ s_lof(e)           ──> flat irregular LoF voids at interfaces + powder (U)
```

Y depends on the process only through U: CT-observable attributes such as type,
count, size, void fraction, severity, layer range and lateral clustering are measured
from the CT label volume, with a detection limit of 4 voxels. Text is then written in
three annotator styles: A detailed, B concise, C structured.

Information that is transferable vs. not transferable from X, which is useful for testing selective alignment:
- **Recoverable from X**: defect type (energy regime), severity, which layers are affected (event layers;
  keyhole pores sit ~2–5 layers below the overheated layer because they form at the keyhole tip).
- **Partly recoverable**: gas porosity (weak H-α line), exact counts (Poisson).
- **Not recoverable**: lateral (x-y) position and clustering, exact pore geometry, LoF orientation details.

## Files (`data/synthetic_v1/`)

| file | content |
|---|---|
| `dataset.h5` | `spectra` (N,16,1024) float32 counts, `wavelength_nm` (1024), `ct` (N,48,64,64) uint8 [z,y,x], 10 µm voxels, `ct_label` (N,48,64,64) 0 matrix/1 keyhole/2 LoF/3 gas/4 powder, `latent/*` (N,16) oracle layer state, `split`, `annotated` |
| `metadata.jsonl` | per region: ids, split, process params, events, nuisances, CT attributes (oracle) |
| `annotations.jsonl` | **learner-visible** text: 10% of train+val regions |
| `test_references.jsonl` | text for all test regions (evaluation only) |
| `annotations_oracle.jsonl` | text for every region (used for annotation-ratio ablations: 1/5/10/20/100%) |
| `config.json` | seed, args, build/part parameters, line list |

Splits are **by build** (7 train / 1 val / 2 test) so test builds have unseen sensor/CT nuisances.
`ct_label`, `latent/*` and `metadata.attributes` are oracle quantities. Use them only for evaluation and
diagnostics, never as model inputs.
