# GEO 384H Final Project

**Recovering the ENSO Attractor from Sea-Surface Temperature Anomalies**
*William Suan · 2 May 2026*

A linear–nonlinear comparison on the canonical climate-physics dataset.
Both PCA and anisotropic diffusion maps are implemented from scratch in
Julia (only `LinearAlgebra.svd` and `LinearAlgebra.eigen` as primitives).

## Headline result

* PC 1 of tropical-Pacific SST anomalies (1950–2024) correlates with
 the official Niño 3.4 index at **ρ = 0.946**, confirming the
 canonical EOF / ENSO equivalence.
* The leading nontrivial diffusion-map coordinate Ψ₂ at α = 1
 recovers the same axis at **ρ = 0.923**, without assuming linearity.
 Both correlations are **43–46 σ** above a permutation null.
* The (Ψ₂, Ψ₃) phase portrait reveals a **curved attractor**: La Niña
 on one end, El Niño on the other, the 1996–99 ENSO cycle traced as
 a coherent loop. PCA's (PC₁, PC₂) plane shows the same data only as
 an unstructured cloud.
* Method-stacking robustness check: Takens-style time-delay embedding **fails** to
 improve the recovery, diagnosed as distance concentration in the
 lifted space + eigenfunction reordering. Reported as a finding.

## Directory layout

```
.
├── README.md      ← you are here
├── Project.toml     ← Julia env (Pkg.instantiate)
├── Manifest.toml
│
├── notebook/ENSO.jl    ← THE main deliverable: Pluto notebook
├── report/
│ ├── plan.md         ← project plan with predictions (pre-registered)
│ ├── report.md       ← written report (markdown source of truth)
│ ├── report.html     ← styled HTML; print → save as PDF
│ └── talk_verbatim.md ← 10-min read-aloud script for the notebook walkthrough
│
├── src/       ← all algorithm code (from scratch)
│ ├── load_sst.jl     ← ERSSTv5 loader + Niño 3.4 fetch
│ ├── pca.jl      ← PCA via SVD + low-rank
│ ├── diffusion.jl    ← anisotropic α-family DMAP + Nyström
│ ├── delay_embed.jl    ← Takens-style time-delay stack
│ ├── lim.jl      ← Linear Inverse Model (Penland & Sardeshmukh)
│ ├── spectrogram.jl    ← Morlet continuous wavelet transform
│ ├── metrics.jl     ← trustworthiness, continuity, Procrustes (utility)
│ ├── enso_analysis.jl   ← end-to-end analysis
│ ├── enso_figures.jl    ← regenerate all 8 figures
│ └── render_report.jl   ← markdown → styled HTML
│
├── data/
│ ├── ersst.v5.sst.mnmean.nc  ← NOAA ERSSTv5 monthly SST (~150 MB)
│ ├── nino34.csv     ← NOAA Niño 3.4 index
│ └── (Salinas .mat files unused; harmless)
│
└── figures/      ← 8 ENSO figures
 ├── fig01_eof1.png       ← EOF 1 horseshoe
 ├── fig02_pc1_nino34.png     ← PC1 vs Niño 3.4 (ρ=0.946)
 ├── fig03_eigenvalues_alpha_scatter.png  ← α-family + eigenvalues
 ├── fig04_phase_portrait.png    ← attractor + 1996–99 loop ★
 ├── fig05_correlations.png     ← method-by-method bars
 ├── fig06_delay_sweep.png     ← method-stacking robustness check
 ├── fig07_null_distribution.png    ← 46σ above null
 ├── fig08_bandwidth.png      ← robustness
 ├── fig09_pc1_spectrogram.png    ← Morlet CWT (Torrence & Compo 1998)
 ├── fig10_lim.png       ← Linear Inverse Model, period 3.1 yr, e-fold 8 mo
 ├── fig11_attractor3d.png     ← 3-D scatter of the climate attractor
 ├── fig12_trajectory_anim.mp4    ← animated phase-portrait trajectory 1990-2010
 ├── fig13_sst_anim.mp4      ← animated SST anomaly map of 1997-98 super-El-Niño
 ├── enso_metrics.txt
 └── lim_summary.txt
```

★ The phase-portrait figure is the project's punchline.

## How to reproduce

```bash
# 1. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Regenerate every figure (~3 min on Apple Silicon)
julia --project=. src/enso_figures.jl

# 3. Re-render the report HTML
julia --project=. src/render_report.jl

# 4. Open the interactive notebook for the live demo
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebook/ENSO.jl")'
```

The ERSSTv5 data and Niño 3.4 index are auto-downloaded on first run.

## Project summary

* **Data:** NOAA ERSSTv5 monthly SST anomalies, 1950–2024, tropical
 Pacific (30°S–30°N, 120°E–80°W). 900 months × 2,322 sea cells.
* **Task:** unsupervised recovery of the dominant climate mode
 ("signal/noise separation" in the segmentation sense, separate ENSO
 signal from background variability).
* **In-course method:** Principal Component Analysis / EOF analysis.
 Lorenz-1956 climate-science canon.
* **Out-of-course method:** Anisotropic Diffusion Maps (Coifman & Lafon
 2006) at α = 1 (Laplace–Beltrami). Recovers manifold geometry.
* **Predictions:** seven, made before running anything (see
 `report/plan.md`). Five pass cleanly, two fail.
* **Validation:** Niño 3.4 index correlation, permutation null
 (n=500), bandwidth sweep, spatial-shuffle baseline.
