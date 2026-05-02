# Project plan, Recovering the ENSO attractor from monthly SST anomalies

**GEO 384H Final · William Suan**

## 1. Scientific question

The tropical Pacific climate state is widely modelled as a low-dimensional
nonlinear dynamical system (Bjerknes 1969; Cane–Zebiak 1985; Jin 1997
"recharge oscillator"). Its dominant slow mode, **El Niño / Southern
Oscillation (ENSO)**, is the largest interannual signal in Earth's
climate system. Operationally, the **Niño 3.4 index** (mean SST anomaly
in 5°S–5°N, 170°W–120°W) is the established 1-D scalar summary of
ENSO state.

If the recharge-oscillator picture is correct, the *full* tropical-Pacific
SST anomaly field at any month should lie close to a low-dimensional
attractor in state space, and the trajectory of monthly snapshots should
trace a curved orbit through that attractor.

**Question.** Can two unsupervised spectral methods, PCA (linear) and
anisotropic diffusion maps (nonlinear), recover this attractor structure
*from monthly SST anomalies alone, without any labels*?

## 2. Data

- **Product.** NOAA ERSST v5 (Extended Reconstructed SST, statistically
 reconstructed from in-situ observations).
- **Source.** NOAA PSL (single netCDF file ≈ 50 MB).
- **Temporal window.** January 1950 – December 2024 (75 years, 900 monthly
 snapshots).
- **Spatial domain.** Tropical Pacific 30°S–30°N, 120°E–80°W (≈ 31 × 80 =
 2 480 grid cells at 2° resolution after masking land).
- **Niño 3.4 index for validation.** NOAA's official monthly Niño 3.4
 index (a 1-D time series), used *only* to score predictions, never as
 input to either algorithm.

## 3. Preprocessing

1. Subtract **monthly climatology** (1950–2024 mean for each calendar
 month at each grid cell) → anomaly field $a(\text{lat, lon}, t)$.
2. Apply **area weighting** by $\sqrt{\cos(\text{lat})}$ to make the
 spatial inner product reflect physical surface area on the sphere.
3. Mask land cells (NaN handling).
4. Reshape to a $T \times N$ data matrix $X$, where $T = 900$ months and
 $N \approx 2{,}480$ valid sea cells. Rows are state vectors; columns
 are spatial locations.

## 4. Methods

Both methods are reused from the existing project, only the data matrix
changes. They are implemented from scratch in Julia, only relying on
`LinearAlgebra.svd` and `LinearAlgebra.eigen`.

### 4.1 In-course: PCA / EOF analysis (Lecture 17)

Take the SVD of the centred $T \times N$ matrix:
$X - \bar{X} = U \Sigma V^\top$. The columns of $V$ are the **Empirical
Orthogonal Functions** (EOFs), spatial patterns. The columns of
$U \Sigma$ are the **Principal Components** (PCs), temporal indices.

This is *the* canonical use of PCA in climate science (Lorenz 1956); the
leading EOF / PC pair is the textbook ENSO mode.

### 4.2a Out-of-course (core): Anisotropic Diffusion Maps on monthly snapshots

Treat each month $t$ as a point $x_t \in \mathbb{R}^N$ in state space.
Build the Gaussian affinity kernel, $\alpha$-renormalise (Coifman–Lafon
2006), row-normalise to a Markov transition matrix $P$, and take its
eigendecomposition. The leading nontrivial eigenvectors $\Psi_2, \Psi_3, \dots$
parameterise the attractor.

At $\alpha = 1$, the embedding asymptotically recovers Laplace–Beltrami
eigenfunctions on the data manifold, independent of sampling density.
Since the climate state is unevenly sampled (more time in neutral than
extreme states), this density invariance is exactly the property we want.

The DMAP problem here is *much smaller* than for hyperspectral: $T = 900$
points vs $N \approx 2{,}480$ features. Full pairwise affinity is
$900 \times 900$, trivial in memory. **No Nyström extension needed.**

### 4.2b Out-of-course (method-stacking robustness check): DMAP on time-delay embeddings

The plain DMAP in §4.2a is *static*: it treats each month as an
independent point and ignores temporal ordering. ENSO is fundamentally a
*dynamical* system, so we test whether adding temporal context improves
the recovery of attractor structure.

By **Takens' embedding theorem** (Takens 1981), a time-delay embedding
of dimension $\ge 2d+1$ is generically diffeomorphic to a $d$-dimensional
attractor of a smooth flow. For ENSO with attractor dimension $d \sim 2{-}3$
(recharge-oscillator picture), a delay dimension of $\sim 7$ months
already suffices; we use $k = 12$ months by default to capture a full
seasonal cycle of context.

We construct the augmented state vectors

$$y_t \;=\; [x_t,\; x_{t-1},\; x_{t-2},\; \dots,\; x_{t-k}] \;\in\; \mathbb{R}^{(k+1)N},\qquad t = k+1, \dots, T,$$

and run the same anisotropic DMAP pipeline on the $y_t$. The leading
nontrivial eigenvectors $\Psi_2', \Psi_3', \dots$ now parameterise the
attractor with full temporal context; consecutive months are
automatically close, so the trajectory through state space is rendered
explicitly.

This connects directly to the neural-manifolds literature, where
delay-coordinate reconstruction is the standard preprocessing for
embedding neural firing-rate trajectories on attractors
(e.g. Mante–Sussillo–Newsome 2013; Pandarinath et al. 2018).

## 5. Predictions (made *before* running anything)

Five concrete, falsifiable predictions:

| # | prediction | falsification criterion |
|:---:|:---|:---|
| **P1** | The leading EOF (PC 1) of tropical-Pacific SST will look like the iconic ENSO horseshoe pattern: a warm tongue along the equator with cooler-than-average flanks. | If EOF 1 looks like a uniform basin warming or other non-ENSO pattern, the canonical PCA/EOF story fails. |
| **P2** | PC 1 (a $T$-vector indexed by month) will correlate with Niño 3.4 at $|\rho| \ge 0.9$ over 1950–2024. | If $|\rho| < 0.8$, ENSO is *not* the leading variance mode and the entire framing is wrong. |
| **P3** | The leading nontrivial diffusion coordinate $\Psi_2$ (plain DMAP, $\alpha = 1$) will correlate with Niño 3.4 at $|\rho| \ge 0.6$. *Honest expectation:* ~0.85, because DMAP doesn't optimise for variance and ENSO is approximately linear at first order. | If $|\rho| < 0.5$ (and PC 1 worked), DMAP is failing to recover the attractor's primary axis even though it exists. |
| **P4** | The DMAP eigenvalue spectrum will show a clear gap after a small index ($k \le 6$), indicating a low intrinsic dimension consistent with the recharge-oscillator picture. | If the spectrum is heavy-tailed with no gap, the low-D attractor hypothesis is empirically rejected. |
| **P5** | In the $(\Psi_2, \Psi_3)$ plane, the strongest historical El Niño months (e.g. Jan 1998, Dec 2015) will appear as outliers far from the bulk of points; the trajectory year-over-year will trace a smooth orbit. | If El Niño months are mixed with neutral months in the embedding, the phase-portrait reading is unsupported. |

P2 is the high-confidence canonical result; passing it just means PCA still
works on SST. P3 is the genuinely *new* test we're running. P5 is the
visual centerpiece for the talk.

### Method-stacking-check predictions (for §4.2b)

| # | prediction | falsification criterion |
|:---:|:---|:---|
| **P6** | DMAP run on time-delay embeddings with $k = 12$ will *improve* the $\Psi_2'$ vs Niño 3.4 correlation by at least 0.05 over plain DMAP, and the $(\Psi_2', \Psi_3')$ phase portrait will show a visibly smoother trajectory between consecutive months. | If correlation does not improve and the trajectory looks no different, the delay embedding adds no value on this dataset. (Negative result.) |
| **P7** | Sweeping the delay $k \in \{0, 3, 6, 12, 24\}$ will show a monotone (or near-monotone) increase in the $\Psi_2'$ vs Niño 3.4 correlation, plateauing in the $k \in [7, 18]$ range, consistent with Takens' bound for a 2–3 dimensional attractor. | If the correlation is flat in $k$ or non-monotone, the dynamical-systems prediction from Takens' theorem is not detectable in this data. |

## 6. Experiments

### E1. PCA / EOF
Compute SVD; plot leading EOF map (P1), PC time series alongside Niño 3.4 (P2).

### E2. Diffusion maps
Compute full pairwise DMAP at $\alpha \in \{0, 1/2, 1\}$. Compare leading
$\Psi_2$ to Niño 3.4 (P3). Plot eigenvalue spectrum (P4).

### E3. Phase portrait
2-D scatter of $(\Psi_2, \Psi_3)$ coloured by Niño 3.4. Sequence of
months connected by lines to show the trajectory. Mark major El Niños
explicitly (P5).

### E4. Robustness
Sweep over $\alpha \in \{0, 1/2, 1\}$, kernel bandwidth scale $\varepsilon$
(median × {0.25, 1, 4}), and time window (1950–2024 vs. 1900–2024).
Show that the qualitative recovery (P3, P5) is stable.

### E5. Negative control
Randomly **shuffle** the time order of monthly snapshots. The SST
anomaly fields are the same, but no temporal ordering remains. PCA's EOFs
will be unchanged (it doesn't see time). Diffusion maps' leading
$\Psi_2$ should still correlate with Niño 3.4 (since DMAP uses the
*static* state-space distances, not time order). This **confirms** that
the manifold structure is encoded in the SST snapshots themselves, not
imposed by the sampling order, closing a potential loophole.

## 7. Validation

- **Trustworthiness / continuity** of the embeddings vs. neighbourhood
 size $k$.
- **Procrustes-aligned bandwidth-stability error** between embeddings at
 different $\varepsilon$.
- **Out-of-sample test**: hold out 2015–2024, embed via Nyström, score
 classification of held-out months as El Niño / Neutral / La Niña using
 thresholds on $\Psi_2$.

## 8. What we cannot claim, and won't

- We do *not* claim DMAP "discovers ENSO from scratch", climate
 scientists already know ENSO is the dominant mode. We claim DMAP
 recovers it *unsupervised*, which is a weaker but verifiable statement.
- We do *not* claim $\Psi_2$ is a *better* index than Niño 3.4. They're
 different things; Niño 3.4 is a fixed regional average, $\Psi_2$ is
 data-driven.
- We do *not* claim the Pacific climate is "actually" low-D in any deep
 sense. The recharge-oscillator picture is a useful approximation; we
 test whether the data supports a low-D embedding, not whether the
 underlying physics is exactly that low-D.

## 9. Tie back to course

- **Lecture 17 (Covariance):** EOF analysis is *the* canonical
 geophysical application of PCA. We re-derive EOFs as the SVD of a
 space–time data matrix.
- **Lecture 18 (Patterns):** truncated EOF reconstruction is a pattern
 filter. We use the same idea in the SST context.
- **Lecture 27 (Time–Frequency):** PC 1 vs Niño 3.4 raises spectral
 questions about ENSO's quasi-periodicity (~3-7 yr). One supplementary
 spectrogram figure.
- **Out-of-course:** the α-family of anisotropic kernels and the
 Laplace–Beltrami $\varepsilon \to 0$ limit, neither of which are
 variance-based methods.

## 10. Headline result the talk should deliver

"The first principal component of tropical-Pacific SST anomalies
correlates with the official Niño 3.4 index at $\rho =$ [number],
recovering the canonical ENSO mode. The first nontrivial diffusion-map
coordinate $\Psi_2$ at $\alpha = 1$ correlates at $\rho =$ [number],
recovering the same axis without assuming linearity. Adding a
12-month time-delay embedding (Takens-style) raises this to
$\rho =$ [number] and renders the climate-state trajectory in the
$(\Psi_2', \Psi_3')$ phase portrait as a visibly smooth orbit; the
strongest historical El Niño months, 1997–98 and 2015–16, appear
as deep, isolated excursions, exactly as the recharge-oscillator picture
predicts."

## 11. Risk register

- *Risk:* Niño 3.4 index file format is annoying. *Mitigation:* it's a
 small CSV, manual download if needed.
- *Risk:* land masking introduces NaNs that break SVD. *Mitigation:* mask
 before reshape, store mask + grid coords for plotting.
- *Risk:* P3 fails (DMAP $\Psi_2$ doesn't correlate with Niño 3.4).
 *Mitigation:* this is *interesting*, not catastrophic. We'd report it
 and discuss what could cause the failure (insufficient
 sampling, over-smoothing, etc.). The whole project is designed to be
 publishable whether predictions pass or fail.

## 12. Time budget

| step | est. hours |
|:---|---:|
| download + load + preprocess data | 1 |
| EOF analysis + PC1/Niño 3.4 plot | 0.5 |
| DMAP + α-family + Ψ₂ vs Niño 3.4 | 0.5 |
| phase portrait + El Niño marker plot | 1 |
| robustness sweep + negative control | 1.5 |
| trustworthiness / continuity / Nyström hold-out | 1 |
| Pluto notebook restructure | 1.5 |
| report rewrite + theory section | 2 |
| presentation outline rewrite | 0.5 |
| **total** | **9.5** |

Existing PCA / DMAP / Nyström / metrics code carries over with no changes.
