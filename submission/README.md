# GEO 384H Final Project — Recovering the ENSO Attractor

**Author:** William Suan
**Date:** 2 May 2026

This folder contains the final project as a self-contained submission.
Everything needed to read the project (PDF) or to reproduce it from
scratch (Julia source) is included.

## What's in this folder

| File / folder | What it is |
|:---|:---|
| `ENSO.pdf` | Static PDF rendering of the Pluto notebook. **Read this if you do not want to install Julia.** Section numbers in the PDF match the notebook one-to-one. |
| `notebook/ENSO.jl` | The live Pluto notebook. This is the project. Prose, code, math, and figures are all in one document. |
| `src/` | The from-scratch Julia implementations the notebook includes. PCA, anisotropic diffusion maps, Linear Inverse Model, time-delay embedding, Morlet wavelet, ERSSTv5 loader. The notebook never reaches outside this folder for algorithmic primitives beyond `LinearAlgebra.svd`, `LinearAlgebra.eigen`, and the matrix logarithm. |
| `figures/` | Static `.png` and `.mp4` assets that the notebook embeds via `PlutoUI.LocalResource`: the 3-D attractor, the Jan-1998 phase-portrait still, the Jan-1998 SST still, and the two animations. |
| `Project.toml`, `Manifest.toml` | Pinned Julia environment for reproducibility. |

## Reading the project (no install)

Open `ENSO.pdf`. It has the full text, every figure, every code cell, and the pass/fail table for the seven preregistered predictions.

## Running the live notebook

You will need Julia 1.12.x (any recent version should work; the manifest pins 1.12.4).

From this folder:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebook/ENSO.jl")'
```

The first command resolves the dependencies (about 1–2 minutes the first time). The second launches Pluto in your browser. The notebook will:

1. Download the NOAA ERSSTv5 NetCDF file (~150 MB) and the NOAA Niño 3.4 CSV on first run, into a `data/` folder it creates next to `notebook/`. Subsequent runs use the cached copies.
2. Run all the methods end-to-end. Total wall time on a 2024 laptop is a few minutes.

## Project structure at a glance

The notebook is organised into nine sections:

* **Section 0** — Setup and library imports.
* **Section 1** — The data: NOAA ERSSTv5 monthly SST, 1950–2024, tropical Pacific, with three preliminary visualisations (sample months, Hovmöller, banded Niño 3.4).
* **Section 2** — Seven predictions made before running anything, with explicit failure thresholds.
* **Section 3** — Method 1: PCA / EOF analysis (in-course).
* **Section 4** — Method 2: anisotropic diffusion maps (out-of-course).
* **Section 5** — Side-by-side method comparison.
* **Section 5.5** — Method 3: Linear Inverse Model.
* **Section 6** — Permutation null negative control.
* **Section 7** — Method-stacking robustness check (negative result).
* **Section 7.5** — Time-frequency check via Morlet wavelet.
* **Section 8** — Pass/fail summary and synthesis.
* **Section 9** — Works cited.

## A note on the data

The raw 150 MB NetCDF file is *not* shipped in this folder. It is downloaded automatically on first run from `downloads.psl.noaa.gov`. If the grader cannot run the notebook live and needs the data file, the URL is hardcoded in `src/load_sst.jl`.
