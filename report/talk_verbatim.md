# Meeting script: expanded overview

**Recovering ENSO-related structure from sea-surface temperature**

> Use the five-minute script for the core talk. This expanded version adds method intuition and figure discussion for a 5–10 minute slot; rehearse to set the actual length. Stage directions are not spoken. Before presenting, replace the contribution cue with one verified sentence about your own work. Do not infer individual ownership from the repository.

## 1. Research question

[Show a monthly SST anomaly map.]

This project asks whether a low-dimensional representation of monthly sea-surface temperature can recover variation associated with ENSO, and what changes when we use diffusion maps instead of PCA.

ENSO is El Niño–Southern Oscillation. The warm phase is El Niño and the cool phase is La Niña. Rather than start with a named index, the analysis starts with a temperature map for each month. A map has thousands of values, but many of those values vary together. The question is whether we can summarize that variation with a few coordinates that have a useful physical interpretation.

[State your verified contribution: what you personally implemented, compared, or interpreted.]

The project is an implementation and comparison of established methods. It is not a new climate model. The concrete workflow in the repository loads the SST fields, constructs anomalies, fits the two representations in Julia, and compares the resulting coordinates with a familiar ENSO reference.

## 2. Data and comparison target

The recorded analysis covers January 1950 through December 2024 in the tropical Pacific. It reports 900 monthly fields and 2,322 retained grid cells. Each row of the input matrix is a month, and each column is a location.

An SST anomaly is a departure from the normal temperature for that calendar month. Removing the average seasonal cycle helps separate interannual variation from the regular annual cycle. In the current workflow, that seasonal reference is calculated over the full sample. The preprocessing also includes weights intended to account for grid-cell area. Data alignment, missing cells, and weight ordering still need technical verification before stronger claims are made.

The comparison target is Niño 3.4, the SST anomaly averaged over a region of the equatorial Pacific. It is not used to fit the embeddings. It is used afterward to align signs, compute correlations, and color the plots.

This distinction matters. The methods do not fit labels, but the reference is not independent of the input: it is another summary of SST. A high correlation says the representation tracks a known SST pattern. It does not establish forecast skill or independent physical validation.

## 3. Why PCA is the baseline

[Show the leading EOF map, then the PC1 time series.]

PCA asks which spatial pattern explains the largest amount of variation in these monthly fields. In climate science, that pattern is called an empirical orthogonal function, or EOF. The corresponding principal component tells us how strongly the pattern appears in each month.

The implementation uses a singular value decomposition of the centered data matrix. That gives a spatial pattern and a monthly coordinate without fitting Niño 3.4 as a target.

The saved results report that PC1 explains about 40 percent of the variance and has a correlation of 0.946 with Niño 3.4. The map and time series are two ways to inspect this result: where the temperature changes occur, and when the pattern is strong or weak.

PCA maximizes explained variance. It does not mathematically maximize correlation with a chosen regional index. The high correlation is an empirical result of this dataset and setup. It makes PCA a strong baseline that a more complicated method has to justify going beyond.

## 4. What diffusion maps changes

[Show the side-by-side representations.]

Diffusion maps begins with similarities between months rather than variance directions. Two months receive a strong connection if their SST anomaly fields are close. These connections form a graph. Eigenvectors of the normalized graph describe patterns of connectivity and provide the coordinates.

One way to picture the method is a random walk between similar monthly fields. That is a mathematical walk on the data graph, not the physical evolution of the climate from one month to the next. Keeping those two meanings separate is important.

There are two main settings in the implementation: a bandwidth that controls how rapidly similarity falls with distance, and a normalization parameter called alpha. The main comparison uses alpha equal to one, which is designed to reduce sampling-density effects under the method's theoretical assumptions. Those assumptions are not a certificate that the finite dataset lies on a smooth physical manifold.

The first nonconstant diffusion coordinate is labeled psi two, because the constant eigenvector is skipped. Its reported correlation with Niño 3.4 is 0.923. Other normalization settings are also recorded in the report, but they do not change the main point: PCA has the higher reported leading-coordinate correlation.

The nonlinear method therefore does not win on this metric. What it adds here is a different organization of the same monthly fields, which can be explored rather than assumed to be better.

## 5. How far to interpret the geometry

[Optionally show the 1996–1999 trajectory, then return to the comparison.]

The diffusion-map view shows a curved arrangement of months. Coloring the points by Niño 3.4 helps locate warm and cool states. Connecting selected months in time gives a trajectory through those fitted coordinates.

The original framing described this as attractor recovery. I would make a narrower claim. A curve or loop in an embedding does not prove a physical attractor, a particular oscillator model, or an intrinsic dimension. Nonlinear coordinates can generate curved plots. PCA can also show curvature when it is present in the projected data.

A useful question is whether the extra diffusion coordinates contain reproducible information beyond the leading SST index. That would need a specified target or stability test, not just a visual preference for one plot.

## 6. Limits and next step

The reported numbers are from the existing project artifacts. They were not regenerated for this text revision.

This analysis compares each month's coordinate with that same month's index. Preprocessing and embeddings use the full record. It is not a forecast and does not measure performance on unseen months.

The original null test shuffled individual months of Niño 3.4. That removes temporal persistence, so its shuffled distribution is not a suitable basis for the original 43–46 sigma significance claims. Those claims should not be used in this presentation.

The next step would be a temporal holdout: learn the seasonal reference and representation on an earlier period, freeze those choices, and project later months without refitting. For diffusion maps, an out-of-sample extension would need to be checked carefully. A dependence-aware null test would address a separate question about how much correlation can arise from persistent time series. These are proposed evaluations, not completed fixes.

## 7. Takeaway

Both methods track ENSO-related variation in the recorded SST sample. PCA is already strong and has the higher correlation. Diffusion maps offers another representation, but the current results do not demonstrate a nonlinear advantage.

That is the main insight I would carry forward: start with the simpler baseline, be explicit about what the comparison measures, and improve the evaluation before adding complexity.

## Optional discussion, not part of the core talk

**Did delays help?** The saved sweep reports lower leading-coordinate correlation when earlier months are stacked with the current month. At 12 lags it is 0.470, compared with 0.923 without delays. That is specific to the setup and metric. Coordinate reordering and changes in distance structure are possible explanations, not proven causes.

**What about wavelets and LIM?** The wavelet view describes frequency content over time. The Linear Inverse Model fits lagged evolution and reports a period and decay time. Both are exploratory backup analyses of the same data, not independent confirmations or held-out forecasts.

**What was your specific contribution?** Answer from your actual work: identify the code, comparison, or interpretation you personally completed, and acknowledge any collaborators or reused components. This script deliberately leaves ownership unassigned.
