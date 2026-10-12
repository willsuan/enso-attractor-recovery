# Recovering ENSO-related structure from sea-surface temperature

GEO 384H project · William Suan

Can a few coordinates summarize ENSO-related variation in monthly sea-surface temperature (SST)? This project compares principal component analysis (PCA), a linear variance-based baseline, with diffusion maps, a nonlinear similarity-based representation. The Julia implementations use SVD and eigendecomposition.

## Existing reported results

The saved analysis covers 1950–2024 in the tropical Pacific and reports:

| Representation | Same-month correlation with Niño 3.4 |
|:--|:--:|
| PCA, first principal component | **0.946** |
| Diffusion maps, first nonconstant coordinate, α = 1 | **0.923** |

These are recorded results in `figures/enso_metrics.txt`, not results reproduced during this prose revision. PCA has the higher reported correlation. The diffusion-map plots offer another view of warm and cool SST states, but their curvature does not establish a physical attractor or a nonlinear advantage.

**Scope:** this is a full-sample, same-month representation analysis, not forecasting or held-out evaluation. Niño 3.4 is a regional SST anomaly index and shares the input's physical basis. It is not used to fit the embeddings, but is used for sign alignment and interpretation. It is a useful reference, not independent evidence of predictive skill.

The original independent-month permutation test breaks temporal dependence. Its reported 43–46 sigma comparisons should not be interpreted as valid time-series significance. A temporal holdout with training-only preprocessing and a dependence-aware null are proposed next steps, not implemented fixes.

## Meeting guide

Use `report/talk_verbatim_short.md` for a roughly five-minute overview, or `report/talk_verbatim.md` for a longer discussion within a 5–10 minute slot. Focus on the question, computational workflow, PCA baseline, diffusion-map comparison, and limitations. Confirm the speaker's specific contribution before presenting; the scripts do not assign individual implementation ownership.

Show the SST/EOF map, PC1 versus Niño 3.4, and the method comparison. Delay embeddings, wavelets, and the Linear Inverse Model (LIM) are optional backup material.

## Files

- `notebook/ENSO.jl`: interactive Pluto notebook, with revised explanations.
- `report/report.md`: current report source.
- `report/report.html`: stale generated report, not regenerated in this revision because Julia was unavailable. Use the revised Markdown instead.
- `src/`: data loading, PCA, diffusion maps, exploratory extensions, and rendering.
- `figures/`: existing figures and recorded metrics. Older figure labels and metric commentary may retain the original attractor/significance language; use the revised report's qualifications.
- `report/plan.md`: historical project plan, not a validation certificate.
- `submission/` and `submission.zip`: unchanged archived submission.

## Running the project

No analysis was rerun for this text revision. These are the existing entry points:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebook/ENSO.jl")'
julia --project=. src/enso_figures.jl
julia --project=. src/render_report.jl
```

The notebook downloads the ERSSTv5 file if missing. The standalone analysis loader expects `data/ersst.v5.sst.mnmean.nc` to exist. The Niño 3.4 loader can download its reference series. Data alignment, masks, and weighting still need technical verification before stronger claims are made.
