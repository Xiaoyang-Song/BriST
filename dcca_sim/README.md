# Circle/crack simulation with annotations

This reuses the simulation from the DCCA monitoring paper (`reference/Source Code`), which is left
unmodified, and adds an annotation layer. Only the simulation data is used here, not the paper's
DCCA monitoring code.

```
python -m dcca_sim.generate --out data/dcca_sim --sigma 1.2 2.0   # images, DGP AE, U, V, spectra, annotations
python dcca_sim/prep.py --sigma 1.2                                 # model-input cache for the BriST pipeline
```

## Data-generating process (as in the reference)
| step | reference source | here |
|---|---|---|
| Y ~ Bern(0.5); Y=0: 1–2 circles (r ∈ [2,6) px); Y=1: 1–2 one-pixel lines (32×32 binary image U) | `notebook/simulation-dgp.ipynb`, `models/simulation.py` | `dgp.generate_images`, with drawn geometry recorded |
| 6-d DGP autoencoder `SimAutoEncoder(6)`, 500 epochs, Adam(1e-3, β=(0.5,0.999)), StepLR(250) | `run_sim_imgae` | `dgp.train_dgp_autoencoder` |
| V = U_latent·β + N(0, 0.1²), β = (0.3, 1, 2, 0.4, −0.5, −0.7) | `jobs/simulation/gen_data.py` | `dgp.make_spectra` |
| X = log\|1/(1 − ReLU(V M) M2)\| + N(0, σ²), 16-d (sp_type `2fc`) | `gen_data.py`, `test.sh` | `dgp.make_spectra`; σ ∈ {1.2, 2.0} |

The process signal X depends on the image only through the 6-d autoencoder code. Defect type
(crack vs. pore) is largely transferable. Finer geometry (location, orientation, exact size) is only
partly transferable, which is the partial-observability setting from the research plan.

## Annotations (new)
Each image is read as a CT cross-section of a 320 × 320 µm region, with 1 px = 10 µm.
`annotate.measure` computes CT-observable attributes from the recorded geometry:

- **pores:** count, diameters, location in a 3×3 grid, overlap, area fraction. Pores-only regions are **acceptable**.
- **cracks:** count, lengths, orientation, location, intersection, whether a crack reaches the edge.
  Severity: **minor** below 100 µm total length, **significant** below 250 µm, **critical** at 250 µm or more or when cracks intersect.

`annotate.describe` writes the text in three annotator styles: A detailed, B concise, C structured.
`dcca_sim/textparse.py` turns any description back into claims and scores it. On the ground-truth
detailed and structured texts it scores 100%.

Splits follow the reference: images 0–4999 are train (4000 train / 1000 val) and 5000–9999 are test.
Expert text is available for 10% of train+val. Text evaluation uses the first 300 val and the
first 1000 test regions, because generation is the expensive step.

## Files (`data/dcca_sim/`)
`images.npy` (N,32,32) · `Y.npy` · `ct_ae-dgp.pt` · `U.npy` (AE latent) · `V.npy` ·
`spectra_sigma{σ}.npy` (N,16) · `metadata.jsonl` (split, annotated, params, attributes, text, annotator) · `config.json`
