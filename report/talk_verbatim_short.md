# Talk script, 10-minute version

**Recovering the ENSO Attractor from Sea-Surface Temperature Anomalies**
GEO 384H · William Suan · 10 minutes

> Tight version of `talk_verbatim.md` for a hard 10-minute slot.
> Same arc, same math intuition, just compressed: figure walkthroughs
> are dropped (the audience can see them on screen as you scroll), and
> the synthesis is a single paragraph.
> Lines preceded by **▸** are stage directions. Do not read aloud.
> Pause at every paragraph break.

---

## 0:00 to 0:45, the hook

▸ Title cell visible.

The tropical Pacific climate state is a twenty-three-hundred-
dimensional vector. But the climate-physics literature, going back
to 1969, has modeled it as a *low-dimensional* dynamical
system, a slow attractor. The dominant mode of that attractor is
ENSO, El Niño–Southern Oscillation.

That's a falsifiable claim about geometry. If the data really live
on a low-dimensional attractor, three different mathematical lenses
should all see the same low-dimensional structure. My project tests
that. The data is monthly sea-surface temperature, SST. Validation
against the official Niño 3.4 index from NOAA, used only as a
yardstick, never as input.

---

## 0:45 to 1:20, the data

▸ Scroll through Section 1: sample months, Hovmöller, banded Niño 3.4.

The data is NOAA's Extended Reconstructed SST version 5, ERSSTv5.
Monthly, 1950 to 2024, 75 years, 900 snapshots. Tropical Pacific
only, about twenty-three hundred valid sea cells.

What you see on screen is what we're decomposing. Three sample
months on the left: the December 1997 super-El-Niño, a near-neutral
month, the December 1999 La Niña. The Hovmöller diagram below
stacks 75 years of equatorial-band SST anomaly vertically: every
El Niño appears as a horizontal red band. The bottom plot is the
validation index, banded by phase, with the strongest events
labeled.

---

## 1:20 to 2:00, predictions

▸ Scroll to the predictions table.

Seven predictions, committed *before* running any of the methods.
The reason: in an exploratory analysis it's easy to look at the
data first and write predictions that fit, which makes "success"
meaningless. Locking down predictions and explicit failure
thresholds in advance turns each pass-or-fail into a real piece of
evidence.

Five core predictions about what the methods should recover. Plus
two method-stacking predictions, P6 and P7, that a Takens-style
time-delay embedding should *improve* the diffusion-maps recovery.
Spoiler: those two will fail.

---

## 2:00 to 2:50, Method 1, PCA

▸ Scroll to Section 3.

PCA, the in-course method, is **the spectral decomposition of the
empirical covariance from Lecture 17**. Each month is a point in
twenty-three-hundred-dimensional state space, and PCA asks which
low-dimensional subspace captures the most variance, equivalently,
which directions the data cloud is most stretched along. Each
direction is an empirical orthogonal function, EOF, a fixed spatial
pattern. The principal component, PC, is the time series of how
strongly that pattern is switched on. Computationally it's the
singular value decomposition, SVD, of the centred data matrix.

▸ Scroll to EOF 1, then PC 1 vs Niño 3.4.

EOF 1 is the iconic ENSO horseshoe, warm equatorial tongue, cooler
subtropical horseshoes flanking it. **P1 passes.** PC-1 against
the official Niño 3.4 index is visually indistinguishable, Pearson
correlation **zero point nine four six**. **P2 passes.**

---

## 2:50 to 4:00, Method 2, DMAP

▸ Scroll to Section 4.

But variance is *linear*. PCA captures the leading axis but not
the *shape* of the cloud around it. The recharge oscillator is a
two-D oscillation, so we expect the climate state to trace a
closed curve in state space, not a Gaussian blob, and we need a
method that can see curvature.

The out-of-course method is anisotropic diffusion maps, DMAP, from
Coifman and Lafon. PCA measures distance with the Euclidean ruler
of the ambient space; DMAP replaces that ruler. It builds a graph
on the data, where edge weights encode "how easy is it to walk
between two months in one step of a random walk on the data?", and
reads geometry off the slow modes of that random walk. The leading
nontrivial eigenvectors of the random-walk transition matrix give
the new manifold coordinates Psi-two, Psi-three, and so on.

One parameter, alpha. Setting alpha to one cancels the
sampling-density bias and recovers the **pure Laplace–Beltrami
operator** on the data manifold. That's the physically right choice
for climate, where most months are near-neutral and the strong
events are rare.

Results: Psi-two correlates with Niño 3.4 at **zero point nine two
three** (**P3 passes**), and the eigenvalue spectrum has a clear
gap at small k (**P4 passes**).

---

## 4:00 to 5:00, the phase portrait

▸ Scroll to "Phase portrait, α = 1".

This is the central figure of the project. Each point is one month,
coloured by Niño 3.4. The cloud is *not* random; it traces a
coherent horseshoe. The strongest historical El Niños sit at the
warm extreme, far from the centroid. Strong La Niñas cluster at the
opposite extreme.

Right panel: I picked the 1996-to-1999 ENSO cycle and drew the
trajectory through the manifold. The state climbs into the warm
phase during 1997, peaks at the December 1997 super-El-Niño in the
lower right, then unwinds through the 1998-99 La Niña along a
*different* path on the manifold. The trajectory is not a closed
circle.

This is Jin's 1997 recharge oscillator, drawn in state space.
**P5 passes.** The curvature is a coordinate-independent statement
that ENSO is nonlinear, and the asymmetric arcs capture the
El Niño / La Niña asymmetry that PCA's PC-2 cannot see.

---

## 5:00 to 5:30, method comparison

▸ Scroll to Section 5, the four-panel scatter.

The same nine hundred months in four embeddings, side by side, all
coloured by the same Niño 3.4 scale. PCA on the left is a
featureless cloud, the variance is one-dimensional. The three DMAP
panels show the horseshoe sharpening as alpha goes from zero to
one. Correlations against Niño 3.4 in the table below are all
between zero point nine two and zero point nine five. Agreement on
the leading axis. The geometry around it is where they disagree,
and that disagreement is the content PCA cannot reach.

---

## 5:30 to 6:30, Method 3, LIM

▸ Scroll to Section 5.5.

PCA and DMAP are *static*: they ignore time-ordering. But the claim
is also that ENSO is a damped oscillator with a roughly three-year
period, and to test that we need a method that uses time-ordering.
That's the Linear Inverse Model, LIM, from Penland and Sardeshmukh.

Assume the state evolves as a linear stochastic ordinary
differential equation, ``dx/dt = L\,x`` plus noise. Estimate the
one-month propagator G of tau directly from data by lag-1
regression. Take the matrix logarithm to get L, eigendecompose.
Each complex eigenpair of L is a damped oscillator with an
explicit period and decay rate. That's the **Lecture 11**
connection: L is the maximum-likelihood estimator of the propagator
under stationary Gaussian noise.

▸ Scroll to the LIM figure.

The least-damped complex eigenpair gives period **three point one
years** and e-folding decay **eight months**. Both numbers are
squarely in the published ENSO literature, and they came out of
the data, not out of fitting to Niño 3.4. The Niño 3.4 correlation
is only zero point six one, expected because Niño 3.4 is a 1-D
scalar while the LIM mode is a 2-D oscillator. The point of LIM is
the period and decay rate, not the correlation.

---

## 6:30 to 6:50, permutation null

▸ Scroll to Section 6.

How do we know correlations of zero point nine two are not just
chance? Shuffle Niño 3.4 in time and re-correlate. Five hundred
shuffles. The shuffle preserves the marginal distribution of the
index and destroys its time alignment. The real correlations are
**forty-six and forty-three sigma** above the resulting null.
Not chance.

---

## 6:50 to 7:40, method-stacking robustness check

▸ Scroll to Section 7.

To stress-test, P6 and P7 were designed *as predictions I expected
to confirm*. Takens 1981: a smooth flow on a d-dimensional attractor
can be reconstructed from a single generic scalar observable, just
by stacking time-delayed copies. So replacing each monthly snapshot
with its delay-vector *should* give DMAP more dynamical context and
improve the recovery. This is the standard rationale for phase-space
reconstruction throughout chaos theory and time-series analysis.

▸ Scroll to the delay-sweep figure.

**It does not.** Across delays from zero to twenty-four months, the
correlation *decreases* monotonically, from zero point nine two
down to zero point one four. **P6 and P7 both fail.** Diagnosis:
distance concentration plus eigenfunction reordering. At twelve
months of lag the ambient dimension is over thirty thousand, the
Gaussian kernel becomes nearly uniform, and ENSO moves off the
leading nontrivial eigenfunction. Naive method-stacking is not free.

---

## 7:40 to 8:15, wavelet temporal check

▸ Scroll to Section 7.5.

One last check. PCA gave the *spatial* mode of ENSO. LIM gave a
global three-year period. The piece in between is checking that
PC-1 *as a time series* has the right local frequency content. A
Morlet continuous wavelet transform, CWT, on PC-1, following
Torrence and Compo. **Lectures 15 and 27 in action**: the wavelet
basis from Lecture 15, the time-frequency representation from
Lecture 27. Implementation from scratch via the fast Fourier
transform, FFT.

Power concentrated in the **two-to-seven year band**, exactly the
canonical ENSO range, strongest excursion at 1997-98. The recovered
mode has the right temporal structure as well as the right spatial
structure.

---

## 8:15 to 9:10, synthesis

▸ Scroll to Section 8, the pass/fail table.

Back to the question. The claim: tropical Pacific climate lives on
a low-dimensional attractor, and ENSO is its dominant mode. Four
methods, none given Niño 3.4 as input, all confirm it: the
**spatial** horseshoe via PCA (**Lecture 17**), the **curved
geometry** via DMAP, the **dynamics** via LIM (**Lecture 11**), the
**temporal** band via the wavelet (**Lectures 15 and 27**). Five of
seven predictions pass; P6 and P7 fail diagnostically.

The methodological point. PCA, diffusion maps, LIM, the wavelet are
not four unrelated techniques. Each is a spectral decomposition of
a different operator: the covariance, the transition matrix, the
propagator, the time-frequency representation. Any structured
sampled dynamical system admits all four, and they answer different
questions about the same data: what shape, what geometry, what
dynamics, what frequency content. That's why these four belong
together as a single bundle.

---

## 9:10 to 9:45, works cited

▸ Scroll to Section 9, works cited.

Coifman and Lafon for diffusion maps. Penland and Sardeshmukh for
LIM. Torrence and Compo for the wavelet treatment. Jin's recharge
oscillator. Takens for the embedding theorem. Plus Lorenz's 1956
EOF analysis. The full list is on screen.

Thank you. I'm happy to take questions.

> The Q&A bank from `talk_verbatim.md` is unchanged and still
> applies; this short script just compresses the read-aloud body.
