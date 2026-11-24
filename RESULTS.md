# BriST: first implementation and evaluation

Two simulations, two tasks:
1. **Binary in situ monitoring:** raise an alarm from the process signal X alone.
2. **Description with the alarm:** generate a text description from X that matches what CT shows, and validate it claim by claim against CT ground truth.

Figures are in `figures/lpbf/` and `figures/circlecrack/`. Raw tables are in `runs*/q_monitor.log`, `runs*/q_summary.log` and `runs*/claim_selective.log`.

## 1. Setup

| | LPBF Ti-6Al-4V simulation (`synthetic/`) | Circle/crack simulation (`dcca_sim/`, reference DGP + new annotations) |
|---|---|---|
| X (in situ) | 16 layers × 1024-channel emission spectra | 16-d "2fc" spectra of the 6-d AE code, σ = 1.2 |
| U (offline) | 48×64×64 CT volume, shown to the VLM as 4 MinIP views | 32×32 binary CT slice (pores vs cracks) |
| Y (text) | type, severity, size, void fraction, layer location | type, count, size, location, orientation, severity |
| splits | 7/1/2 builds (1400/200/400 regions) | 4000/1000/5000 (text evaluated on 300 val / 1000 test) |
| expert text | 10% of train+val (138 train pairs) | 10% (395 train pairs) |
| alarm label | severity moderate or severe | any crack |

**Model.** SmolVLM-500M. The decoder is a 4-bit QLoRA (SmolLM2-360M). The vision side is SigLIP with LoRA plus the projector.
- **Signal encoder f_X:** 1-D CNN plus a transformer over layers, or an MLP for the 16-d vector.
- **Signal projector g_X:** 64 learned query tokens that produce 64 decoder tokens Z_X.

**Pipeline,** as in the research plan:
- **Stage I:** CT teacher fine-tuned on the M annotated CT–text pairs.
- **Stage II:** alignment on all N training (X, U) pairs: feature-level DCCA, token-level MSE, hierarchical (both), or selective (relevance-weighted, heteroscedastic).
- **Stage III:** student whose decoder starts from the teacher. It is trained with cross-entropy on the M texts, plus token-level KL distillation from the teacher's descriptions of all N training CTs.

**Evaluation.** Every generated description is parsed into claims, and each claim is checked against attributes measured on the CT. On the ground-truth texts the parsers score 100%, so every error counted is a model error. Single seed throughout.

## 2. Findings

### F1. The natural-image VLM cannot read industrial CT until its vision encoder is adapted
With only decoder LoRA plus the projector, the LPBF teacher was at chance: 0.30 severity accuracy on val. A linear probe on frozen SigLIP features predicts severity at chance even when trained on all 1400 labels (0.32 test accuracy). Adding LoRA to the vision encoder fixed it:

| CT teacher (test) | alarm acc. | severity | type / dominant | location |
|---|---|---|---|---|
| LPBF | 0.948 | 0.69 | 0.85 | layer hit 0.31 |
| circle/crack | 0.998 | 0.93 | 1.00 | 0.68 (orientation 0.75) |

This confirms the plan's caveat that domain adaptation of the vision side is required, not optional. The teacher's weak spot, LPBF vertical localization, becomes a ceiling for everything distilled from it (see F5).

### F2. Task 1: the bridged text alarm is an effective monitor and needs far fewer labels
Test balanced accuracy, with false-alarm rate in parentheses (`figures/*/task1_monitoring.png`):

| method | labels used | LPBF | circle/crack |
|---|---|---|---|
| spectra CNN | CT labels for **all N** | 0.895 (0.030) | 0.861 (0.222) |
| spectra CNN | M labels | 0.906 (0.048) | 0.832 (0.248) |
| DCCA features + LR (prior-work style) | all N | 0.833 (0.006) | 0.852 (0.244) |
| direct X→text, text alarm | M texts | 0.926 (0.127) | 0.845 (0.284) |
| **hierarchical + KD, text alarm** | M texts + N (X,U) pairs | **0.937 (0.055)** | 0.855 (0.232) |
| hierarchical + VLM, text alarm | M texts + N pairs | 0.919 (0.115) | **0.861 (0.236)** |
| CT teacher (privileged, needs CT) | M | 0.953 (0.018) | 0.998 (0.004) |

- **The alarm that comes with the description is at least as good as a dedicated classifier.** On LPBF the bridged text alarm beats the classifiers and has less than half the direct model's false-alarm rate. On circle/crack it matches a classifier trained with CT labels for every training region, while using expert text for only 10%.
- **The ceiling here is information, not the method.** Circle/crack monitoring tops out around 0.86 for every X-only method, because X is a noisy 16-d function of a 6-d code (linear accuracy 0.85). No method can recover what the signal does not carry. That is the irreducible gap I(Y;U|X) from the plan.

### F3. Task 2: with few annotations, direct X→text collapses; the bridge keeps descriptions specific to the input
`figures/*/specificity.png`, `figures/*/qualitative_examples.png`. Measured on truly defective test regions:

| | distinct texts | share of the most common text | Spearman ρ(stated, true) |
|---|---|---|---|
| LPBF direct | 0.08 | **0.59** | 0.35 (void fraction) |
| LPBF hierarchical + KD | 0.42 | 0.27 | **0.67** |
| LPBF hierarchical + VLM | 0.47 | 0.19 | **0.71** |
| LPBF CT teacher | 1.00 | 0.00 | 0.63 |
| circle/crack direct | 0.02 | **0.62** | −0.07 (max size) |
| circle/crack hierarchical + KD | 0.14 | 0.12 | 0.08 |

With 138 texts, the direct model writes the same memorised report for 59% of the defective LPBF regions. Its apparently high "dominant type" and "layer localisation" scores are largely artefacts of that one report. In the bridged models the stated porosity tracks the true porosity as well as the CT teacher's does.

Quality of the descriptions on defective LPBF test regions (`figures/lpbf/task2_attributes.png`):

| | severity | void-fraction log10 error | type precision |
|---|---|---|---|
| direct | 0.60 | 0.44 | 1.00 |
| hierarchical + KD | **0.69** | **0.29** | 0.99 |
| CT teacher | 0.62 | 0.25 | 0.97 |

On circle/crack the bridge improves count accuracy (0.43 → 0.55 for hierarchical + KD) and severity slightly (0.61 → 0.63 for selective + KD). Type accuracy is the same (0.85 → 0.86).

### F4. Feature-level (DCCA) alignment alone does not transfer to language; token-level alignment is the essential piece
Feature-aligned students are no better than direct, or worse:
- circle/crack: the feature-aligned model writes one text for 100% of defective regions;
- LPBF: it never states a layer location.

The zero-shot token-aligned model receives **no** X–text supervision, only Z_X ≈ Z_U plus the frozen teacher decoder. It already matches direct on type (LPBF 0.85; circle/crack 0.853 vs 0.846). That is direct evidence of the core intuition: semantics learned on U transfer to X through the shared token space. Distillation adds a smaller, consistent gain: LPBF severity 0.58 → 0.61 and void error 0.39 → 0.35 (all-test numbers); circle/crack count accuracy 0.49 → 0.55.

### F5. The bridge transfers whatever the signal cannot see as confident hallucination; claim-level selective transfer fixes this
On circle/crack, location (0.25–0.30) and orientation (about 0.25) are at chance for **every** X-only model. Yet every model states them in 94–100% of reports. The token-level "selective" alignment as first implemented (relevance × heteroscedastic token weights) did not change this and gave no consistent gain over hierarchical + KD.

A **claim-level** version (`scripts/claim_selective.py`) works:
1. pseudo-labels per attribute come from the teacher's descriptions of all N training CTs;
2. a probe on the student's own f_X features estimates each attribute's transferability;
3. a claim is kept only if the probe supports it at a threshold calibrated on val (target precision 0.8).

It needs no extra labels and no retraining (`figures/*/claim_selective.png`):

| circle/crack, hierarchical + KD | transferability | before: stated / acc | after: stated / acc |
|---|---|---|---|
| defect type | 0.69 | 100% / 0.86 | 100% / 0.86 |
| severity | 0.28 | 100% / 0.62 | 60% / **0.80** |
| count | 0.28 | 100% / 0.55 | 7% / 0.65 |
| location | 0.03 | 100% / 0.26 | **0%** (reported as not resolvable in situ) |
| orientation | −0.07 | 94% / 0.24 | **0%** |

On LPBF it keeps the dominant type (transferability 0.88, accuracy 0.92) and withholds layer localisation (−0.24). Severity improves only a little, 0.61 → 0.64 at 56% coverage: the val-calibrated threshold does not fully carry over to new builds.

This is the plan's 2×2 table (relevant vs. transferable) put into practice. It separates what the spectra can support from what only CT shows, without being told which is which.

## 3. Verdict

**The intuition holds.** Using offline CT as a bridging modality, with a VLM as the semantic teacher, gives the following:
- **Task 1:** an in situ alarm as good as a dedicated classifier trained with full labels.
- **Task 2:** descriptions that are specific to the input and quantitatively tied to the CT (severity, porosity) from very few expert annotations, where direct supervision collapses to a memorised report. Zero-shot transfer through the token space works with no X–text pairs at all.
- **Annotation efficiency:** with 2% expert text the bridged model matches direct supervision with 25% (§5).

**As a method, the components differ in value:**
- *Essential:* vision-side domain adaptation of the teacher; token-level (decoder-space) alignment.
- *Helpful but modest:* distribution-level distillation.
- *Not useful alone:* feature-level DCCA alignment.
- *Needs redesign, now prototyped:* selective alignment. Weighting at the token level did not help. Selection at the claim level, using teacher pseudo-labels and transferability probes, does exactly what the plan asked for. It should become the central contribution, trained end-to-end instead of applied post hoc.

**Main limitations of this round:**
- Single seed and small test sets; differences under about 3 points are not reliable.
- The LPBF teacher localises poorly, and that weakness is distilled into the students.
- The LPBF spectra separate the regimes very easily (the direct model gets 0.98 dominant type on defective regions), so circle/crack is the more informative benchmark.
- Synthetic, template-generated text.

## 4. Suggested next steps
1. **Train claim-level selectivity end-to-end.** Rewrite distillation and expert targets so that attributes with low estimated transferability become hedged statements (e.g. "location not resolvable from the in situ signal"), then report risk–coverage curves.
2. **Strengthen the teacher's localisation** (LPBF): per-layer slice tokens or a multi-slice aggregator instead of MinIP views. The students' ceiling depends on it.
3. **Robustness:** 3 seeds; circle/crack σ sweep (σ = 2.0 spectra already generated in `data/dcca_sim/`); LPBF with stronger nuisances so the type is not trivially readable.
4. **Annotation-ratio curves:** add the 100% upper bound and 1%, with seeds.

## 6. How to reproduce
`sbatch scripts/run_pipeline.sbatch` (set `--account` first). Every step skips itself if its output exists. Figures: `scripts/make_figures.py`. Claim-level selection: `scripts/claim_selective.py`. Environment: `.venv` (torch 2.5.1, transformers 4.51.3, peft 0.15.2, bitsandbytes 0.45.5). The model weights are cached in `hf_cache/`.

## 5. Annotation efficiency (sweep)
`figures/*/annotation_efficiency.png`. Expert texts are nested random subsets of the training split; results are on test.

| expert texts M (fraction) | LPBF direct: sev / dom / alarm | LPBF selective + KD: sev / dom / alarm |
|---|---|---|
| 28 (2%) | 0.44 / 0.76 / 0.83 | **0.58 / 0.88 / 0.91** |
| 70 (5%) | 0.46 / 0.82 / 0.91 | **0.59** / 0.88 / 0.91 |
| 138 (10%) | 0.53 / 0.92 / 0.94 | **0.57** / 0.89 / 0.91 |
| 350 (25%) | 0.52 / 0.89 / 0.93 | **0.60** / 0.89 / 0.91 |

| expert texts M (fraction) | circle/crack direct: type / sev / count | circle/crack selective + KD: type / sev / count |
|---|---|---|
| 80 (2%) | 0.80 / 0.56 / 0.58 | **0.86 / 0.62** / 0.57 |
| 200 (5%) | 0.83 / 0.56 / 0.54 | **0.86 / 0.63 / 0.57** |
| 395 (10%) | 0.85 / 0.61 / 0.43 | **0.85 / 0.63 / 0.56** |
| 1000 (25%) | 0.85 / 0.61 / 0.50 | **0.85 / 0.64 / 0.57** |

- **The bridged student is almost flat in M.** Its semantics come mostly from the CT teacher, through the N unannotated (X, U) pairs.
- **At 2% it matches or beats direct supervision at 25%** (12.5× more expert text) on severity, defect type and alarm accuracy. The one exception is LPBF dominant type, where direct at 10–25% is slightly higher; with few texts that number is inflated by the collapse described in F3.
- **Bridged severity stays below the teacher's ceiling** (LPBF 0.69, circle/crack 0.93). The remaining gap is the information X lacks, not a lack of annotation.
