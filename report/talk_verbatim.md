# Talk script, verbatim read-aloud

**Recovering the ENSO Attractor from Sea-Surface Temperature Anomalies**
GEO 384H · William Suan · 10 minutes

> The presentation IS the Pluto notebook (`notebook/ENSO.jl`). There are
> no slides. As you read this script aloud, you scroll the notebook from
> top to bottom; the script is timed and worded to match what's on
> screen at each section.
> Lines preceded by **▸** are stage directions. Do not read them aloud.
> Course-lecture pointers (Lecture 17, etc.) live only in this script,
> not in the notebook itself; mention them out loud as you walk through.
> Pause briefly at every paragraph break.

---

## 0:00 to 0:45, Title cell, the hook

▸ Notebook position: top of the notebook, the title cell visible.

The tropical Pacific climate state is a 2,322-dimensional vector,
one number per grid cell of sea-surface temperature. But going back
to Bjerknes in 1969, the climate-physics literature has modeled it
as a *low-dimensional dynamical system*: a slow attractor with
basically one or two macroscopic degrees of freedom. The dominant
mode of that attractor is ENSO, El Niño–Southern Oscillation.

That is a falsifiable claim about geometry. If the climate state
genuinely lives on a low-dimensional attractor, three different
mathematical lenses should all see the same low-dimensional
structure inside the high-dimensional data. **My project tests
that.** Monthly SST anomalies, 1950 to 2024, validation against
NOAA's official Niño 3.4 index, used only as a yardstick, never
as input.

The three lenses, listed at the top of the notebook. PCA, the
in-course method from **Lecture 17 on Covariance**, recovers the
spatial mode. Anisotropic diffusion maps, the out-of-course method,
recovers the curved manifold geometry. The Linear Inverse Model from
Penland and Sardeshmukh, the **Lecture 11 estimation** connection,
recovers ENSO's dynamical period and damping rate. Plus a Morlet
wavelet spectrogram from **Lectures 15 and 27** to confirm the
temporal structure.

All algorithms are written from scratch in Julia. The only primitives
I take from a library are `LinearAlgebra.svd`, `LinearAlgebra.eigen`,
and the matrix logarithm.

---

## 0:45 to 1:45, Section 1, Data

▸ Scroll to "## 1, Data: NOAA ERSSTv5 monthly SST".

The data is NOAA's ERSST version five, monthly sea-surface
temperature, reconstructed from in-situ observations, 1950 to 2024.
That's 75 years, 900 monthly snapshots.

I restricted to the tropical Pacific, thirty south to thirty north,
one-twenty east to eighty west, at two-degree resolution. About
twenty-three hundred valid sea cells.

For validation, I use NOAA's official Niño 3.4 index. To be clear:
this is validation only. It is never an input to any of the three
methods.

▸ Scroll past the data-loading cells to the "Sample months" figure.

Three monthly snapshots of the data we're working with: the December
1997 super-El-Niño on the left, a near-neutral January 1989 in the
middle, and the December 1999 La Niña on the right. Same colormap,
same range, dashed Niño 3.4 box on each. **This is what one row of
our T-by-N data matrix looks like when reshaped onto the lat-lon
grid.**

▸ Scroll to the Hovmöller diagram.

Equatorial-band SST anomaly through time, longitude on the x-axis,
year on the y-axis, 75 years stacked vertically. **One picture
summarises the data we're trying to recover.** Every El Niño appears
as a horizontal red band, the 1972, 1982-83, 1997-98, 2015-16, and
2023 events labeled. La Niñas appear as cool blue bands. The
westward-marching warm tongue and the eastern-Pacific peak are both
visible.

▸ Scroll to the banded Niño 3.4 figure.

The validation index, NOAA Niño 3.4, banded by ENSO phase: red fills
above zero, blue fills below. Dashed thresholds at plus and minus
zero point five degrees mark El Niño and La Niña classification.
The five strongest events stand head and shoulders above the
threshold.

---

## 1:45 to 2:15, Section 2, Predictions

▸ Scroll to the predictions table, "## 2, Predictions made before
running anything".

I committed seven predictions *before* running anything. The reason:
in an exploratory analysis it's easy to look at the data first and
then write predictions that fit, which makes "success" meaningless.
Locking down predictions and explicit failure thresholds in advance
turns each pass-or-fail into a real piece of evidence.

Five core predictions. **P1**: EOF-1 should look like the ENSO
horseshoe. **P2**: PC-1 should correlate with Niño 3.4 above zero
point nine. **P3**: the diffusion-map coordinate Psi-two should
correlate above zero point six. **P4**: the DMAP eigenvalue spectrum
should have a clear gap. **P5**: major historical El Niños should
appear as far outliers in the phase portrait.

Plus two method-stacking predictions, **P6** and **P7**: a
Takens-style time-delay embedding should *improve* the DMAP recovery,
and the improvement should be monotone in the lag. Spoiler: those
two will fail.

---

## 2:15 to 3:00, Section 3, Method 1 PCA

▸ Scroll to "## 3, Method 1: Principal Component Analysis (in-course)".

The in-course method is principal component analysis, **the spectral
decomposition of the empirical covariance operator from Lecture 17**.
Each month becomes a point in twenty-three hundred dimensional state
space. PCA asks which low-dimensional linear subspace captures the
most variance.

The construction, written out in the notebook: take the SVD of the
centered space-time data matrix. Columns of V are EOFs, spatial
patterns. Columns of U-times-Sigma are principal components, temporal
patterns. By Eckart-Young, the rank-k truncation is the optimal
rank-k approximation of X in any unitarily-invariant norm.

▸ Scroll to the "ρ(PC1, Niño 3.4) = 0.946 → P2 PASS" badge.

The output line at the top of the cell shows the correlation:
**zero point nine four six**, P2 passes with margin.

---

## 3:00 to 4:00, EOF 1, PC 1, and EOFs 2-4

▸ Scroll to the EOF 1 figure (subsection "EOF 1, the leading spatial
mode"). The map fills the cell.

This is exactly the iconic ENSO horseshoe. A warm equatorial tongue,
deep blue, extending from the central Pacific to the South American
coast. Cooler subtropical horseshoes flanking it. The Niño 3.4 box,
dashed, sits right inside the warm tongue. **P1 passes.**

▸ Scroll to the PC 1 vs Niño 3.4 figure ("### PC 1 vs the Niño 3.4
index").

PC-1 over time, overlaid with the official Niño 3.4 index. The two
curves are visually indistinguishable. The Pearson correlation is
**zero point nine four six**, over 900 months from 1950 to 2024.

The black stars mark the strongest historical El Niños, '72, '82,
'97, 2015, 2023. Every one of them appears as a peak in both curves.

This is **Lecture 18, Patterns**, in action: the leading EOF is a
literal pattern of the leading climate mode.

▸ Scroll to the four-panel "Higher-order EOFs" figure.

Briefly, EOFs two through four. EOF 1 captured forty percent of the
variance. EOF 2, fifteen percent, is a Pacific Decadal Oscillation-like
mode. EOF 3, eight percent, has a different equatorial structure
sometimes called the Modoki pattern. EOF 4, four percent, is
mid-latitude. The point: PCA does see structure beyond ENSO, just
nothing else of comparable amplitude.

---

## 4:00 to 4:45, Section 4, Method 2 DMAP

▸ Scroll to "## 4, Method 2: Anisotropic Diffusion Maps (out-of-course)".

So PCA gave us the linear answer: yes, ENSO is the leading variance
mode. But variance is a *linear* notion. PCA always returns ellipsoidal
clouds because its only knob is "biggest direction." If the climate
dynamics are nonlinear, PCA captures the leading axis but not the
*shape* of the cloud around it. The recharge oscillator is a 2-D
oscillation, so we expect a closed curve in state space, not a
Gaussian blob, and we need a method that can see curvature.

The out-of-course method is anisotropic diffusion maps, from Coifman
and Lafon, 2006.

The intuition: PCA measures distance with the Euclidean ruler of the
ambient space. Diffusion maps replaces that ruler. It builds a graph
on the data, where edge weights encode "how easy is it to walk
between two months in one step of a random walk on the data?", and
reads geometry off the slow modes of that random walk. Two months are
*close* in the diffusion-map embedding if the random walk visits them
at similar transition probabilities — a notion of distance that
respects manifold geometry rather than ambient Euclidean structure.

The construction in three steps. Build a Gaussian similarity kernel
``k_\varepsilon`` with bandwidth epsilon. Alpha-renormalize to remove
sampling-density bias. Row-normalize to get a Markov transition matrix
``P``. The leading nontrivial eigenvectors of ``P`` become the new
manifold coordinates ``\Psi_2, \Psi_3, \dots``.

▸ Scroll to the alpha-family table and to the "Why α = 1?"
subsection.

The alpha parameter is the construction's defining feature. As
epsilon shrinks, alpha equals zero recovers the graph Laplacian,
which is biased by where the data sits densely. Alpha equals one-half
recovers the Fokker–Planck operator with a drift in the
log-probability. Alpha equals one recovers the **pure Laplace–Beltrami
operator on the data manifold**, sampling-density-invariant.

That last one matters for climate. Most months are neutral, the
strong El Niños are rare. We don't want the embedding to be biased
toward the centroid where the data piles up — we want the manifold
of *possible* states. So **alpha equals one is the physically
motivated choice**, and we use it as the primary value.

---

## 4:45 to 5:00, DMAP slider and α-family

▸ Scroll to the alpha selector and diffusion-time slider. Show the
"Selected α = 1.0, t = 1" line below.

The selector lets us pick alpha live; we leave it at one. The slider
controls diffusion time t, raising eigenvalues to the t-th power.
Default t equals one.

Below the scatter the cell shows the live correlation badge: rho of
Psi-two against Niño 3.4 is **zero point nine two three**, matching
PCA. **P3 passes.**

---

## 5:00 to 5:15, DMAP eigenvalue spectrum

▸ Scroll to "### Eigenvalue spectrum (P4)".

A clear gap in the eigenvalue spectrum: lambda 2 around zero point
four seven, well separated from the trivial lambda 1. The eigenvalues
decay smoothly through about five slow modes before steeply dropping,
consistent with the recharge-oscillator-plus-Modoki count from the
climate-dynamics literature. **P4 passes.**

Note this is something **PCA's variance scree cannot give us**, the
DMAP eigenvalue spectrum has a kink, and it counts modes.

---

## 5:15 to 5:45, Kernel matrix and nearest neighbors

▸ Scroll to "### What the kernel actually sees: the affinity matrix".

This is the nine hundred by nine hundred pairwise affinity matrix
K-epsilon, ordered chronologically on both axes. Bright off-diagonal
patches are pairs of months in *different decades* whose SST patterns
look similar. The leading eigenvectors of the row-normalised version
of this matrix become Psi-two, Psi-three, and so on. **This is the
literal object diffusion maps is reading geometry off of.**

▸ Scroll to "### What does similar to Dec 1997 actually mean?".

A pedagogical figure. I picked December 1997, the super-El-Niño peak,
and pulled the ten months whose SST patterns are most similar to it
under the Gaussian kernel. They cluster tightly in the warm corner of
the manifold. Eight of them are 1997-98 months, two are December '82
and January '83, the prior super-El-Niño. **The kernel encodes "two
months in similar climate configurations have high affinity."**

---

## 5:45 to 6:30, Phase portrait, P5

▸ Scroll to "### Phase portrait, α = 1 ... and the (Ψ₂, Ψ₃)
horseshoe". Both panels visible.

This is the central figure of the project, and the answer to the
"can DMAP show curvature" question.

**Left panel.** Each point is one month, coloured by Niño 3.4. The
cloud is not random; it traces a coherent horseshoe. The strongest
historical El Niños, December '72, '82, '97, 2015, 2023, sit at the
warm extreme of the manifold, far from the centroid. Strong La Niñas
cluster at the opposite extreme. **The Niño 3.4 colour gradient
follows Psi-two**, which is what gives us the 0.92 correlation,
*and* the cloud is curved in the perpendicular direction, which is
what PCA cannot see.

**Right panel.** I picked one four-year ENSO cycle, January 1996
through December 1999, and drew the trajectory through the manifold.
The climate state climbs into the warm phase during 1997. It peaks
at the December 1997 super-El-Niño in the lower right of the
embedding. Then it unwinds through the strong 1998–99 La Niña along
a **different path** on the manifold. The trajectory is not a circle.

This is the **recharge oscillator**, the leading dynamical model of
ENSO from Jin 1997, *drawn in state space*. **P5 passes.** The
geometric content here is twofold: the curvature is a coordinate-
independent statement that ENSO is genuinely nonlinear, and the
fact that El Niño and La Niña arcs are different paths captures the
well-known ENSO asymmetry that PCA's PC-2 cannot see (Burgers et al.
2005).

---

## 6:30 to 7:00, Density contours and seasonal phase-locking

▸ Scroll to "### Density contours: where does the climate state spend
its time?".

The same Psi-two, Psi-three plane, with a two-dimensional occupancy
estimate plotted as filled contours behind the points. Most of the
climate state's time is spent in the central neutral region of the
manifold. The warm El Niño and cool La Niña corners are visited
rarely. **The shape of the high-density region is the climatological
attractor in the Lebesgue sense, literally where probability piles up
in the embedding.**

▸ Scroll to "### Trajectory by month-of-year".

The 1996-99 trajectory again, but this time coloured by month of year
on a circular colormap. Notice how the climate state hits the warm
corner in late boreal autumn, early winter each year. **This is the
seasonal phase-locking of ENSO peaks**, ENSO events tend to mature
between November and January, and the manifold makes it visible.

---

## 7:00 to 7:15, 3-D attractor and the 1997-98 peak

▸ Scroll to "### 3-D attractor and the 1997-98 peak". Show Figure 11.

In three diffusion-map coordinates the attractor opens into a curved
two-dimensional sheet wound through three-dimensional volume. The
fourth coordinate Psi-four spreads out the El-Niño side along an axis
the two-dimensional projection collapses.

▸ Scroll past Figures 12 and 13, the Jan 1998 peak stills.

Two stills showing the climate state at the January 1998 peak of the
super-El-Niño, in the manifold (Figure 12) and in raw spatial form
(Figure 13). The full animations live in `figures/`; in the manifold
view the climate state has migrated to the warm corner, and in the
spatial view the warm tongue stretches across the entire central-to-
eastern equatorial Pacific. **The same climate state, two coordinate
systems.**

---

## 7:15 to 7:45, Section 5, side-by-side method comparison

▸ Scroll to "## 5, Method comparison". The four-panel scatter is
visible.

To make the contrast between the two methods literal, here are the
same nine hundred months in the four embeddings, side by side, all
coloured by the same Niño 3.4 scale.

PCA's PC-1 versus PC-2 plane on the left: a featureless cloud. All
the variance is along PC-1, the second is just orthogonal noise. The
three diffusion-map panels show the same data as a *curved manifold*,
and the curvature gets cleaner as alpha goes from zero to one. At
alpha equals one, the horseshoe is unambiguous, La Niña on the cool
end, El Niño on the warm end.

**Both methods agree on the leading axis** — the correlations with
Niño 3.4 in the table below are all in the 0.92 to 0.95 range. **They
disagree dramatically on the geometry around that axis.** That
disagreement is the geometric content the in-course method cannot
reach.

---

## 7:45 to 8:45, Section 5.5, Linear Inverse Model

▸ Scroll to "## 5.5, Method 3: Linear Inverse Model (LIM)".

Both PCA and diffusion maps are *static* state-space methods. They
treat each month as an independent point and ignore time ordering.
But the climate-physics claim isn't just "ENSO has a spatial pattern
and a curved manifold." It's also that **ENSO is a damped oscillator
with a roughly three-year period**. To check that, we need a method
that *uses* the time ordering. That's the Linear Inverse Model from
Penland and Sardeshmukh, 1995.

The construction. Assume the climate state evolves as a stationary
stochastic linear ODE, ``dx/dt = L\,x + Q\,\xi(t)``. Estimate the
one-month propagator ``G(\tau) = C(\tau)\,C(0)^{-1}`` directly from
data, take the matrix logarithm to get ``L``, then eigendecompose
``L``. Each complex eigenpair of ``L`` is a damped oscillator with an
**explicit period** ``T = 2\pi/|\omega|`` and **e-folding decay**
``\tau_d = -1/\alpha``. This is a textbook linear-state-space
estimation problem, the **Lecture 11 connection**: ``L`` is the
maximum-likelihood estimator of the linear propagator under
stationary Gaussian noise.

▸ Scroll down to the LIM eigenvalue-and-time-series figure.

I fit LIM on the top ten PC scores at lag one month. The least-damped
complex eigenpair gives **period three point one years and e-folding
decay zero point six eight years**, about eight months. Both numbers
are squarely in the published-LIM literature range for ENSO. **The
key point: those numbers came out of the data, not out of fitting
to Niño 3.4.**

The right panel: the LIM mode time series overlaid with Niño 3.4.
Correlation is zero point six one, lower than PCA's zero point nine
five. That's expected. Niño 3.4 is a one-dimensional scalar; LIM's
natural representation is a *two-dimensional* oscillator (the complex
eigenpair has an in-phase and a quadrature component). Projecting
onto one axis loses half the information. **The point of LIM is not
the correlation, it's the explicit period and decay rate** that the
static methods cannot give.

---

## 8:45 to 9:05, Section 6, Permutation null

▸ Scroll to "## 6, Negative control: permutation null".

At this point we have three methods all reporting correlations above
zero point nine. That **sounds** impressive, but the obvious worry
is: maybe SST data is just spatially correlated enough that *any*
decomposition will pick up some low-frequency mode that aligns with
a heavy-tailed climate index. Maybe 0.92 is what you'd get by chance.

The cleanest check: shuffle Niño 3.4 in time and re-correlate. Five
hundred shuffles each. The shuffle preserves the marginal distribution
of the index, same range, same variance, same fat tails, while
destroying its time alignment with PC-1 / Psi-two. **Anything above
this null distribution is real recovered structure, not artefact.**

The null distributions concentrate around ``\rho = 0.03``. The real
correlations are **forty-six and forty-three standard deviations
above** the null. The recovery is not chance.

---

## 9:05 to 9:35, Section 7, Method-stacking robustness check

▸ Scroll to "## 7, Method-stacking robustness check: time-delay
diffusion maps".

So far everything has worked. To stress-test that, I designed P6 and
P7 *as a prediction I expected to confirm*. The premise: a common
pattern in applied dimensionality reduction is to **stack methods**,
preprocess with one technique then apply another. The particular
stack I tested is Takens' embedding theorem followed by diffusion
maps.

**Takens 1981** is a remarkable theorem: a smooth flow on a
``d``-dimensional attractor can be reconstructed from a single
generic scalar observable, just by stacking time-delayed copies. So
plausibly, replacing each monthly snapshot ``x_t`` with the
delay-vector ``[x_t,\, x_{t-\tau},\, \dots,\, x_{t-k\tau}]`` should
give DMAP more dynamical context per point and *improve* the
recovery. This is also the standard rationale for delay embeddings
in computational neuroscience (Mante–Sussillo, Pandarinath), so I
had a personal stake in seeing it work.

▸ Show the delay-sweep figure.

**It does not.** Across delays from zero to twenty-four months, the
leading-mode correlation *decreases* monotonically, from zero point
nine two down to zero point one four. **P6 and P7 both fail.**

Two contributing factors. **One: distance concentration.** At twelve
months of lag, the ambient dimension is over thirty thousand, and
the ratio of maximum-to-median pairwise distance collapses from 6.9
to 2.0. That's the classical curse of dimensionality, and it makes
the Gaussian kernel nearly uniform — the diffusion map has nothing
left to discriminate on. **Two: eigenfunction reordering.** Each
added lag introduces new low-frequency modes (the annual cycle,
lagged seasonal correlations) that displace ENSO from being the
leading nontrivial eigenfunction. The signal isn't lost, it just
isn't on Psi-two anymore.

**The lesson: naive method-stacking is not free**, and plain DMAP
suffices for ENSO. This is exactly the kind of negative result that
makes the preregistered predictions worth doing — if I'd written
P6 *after* running the analysis, I'd never have included it.

---

## 9:35 to 9:55, Section 7.5, Wavelet spectrogram

▸ Scroll to "## 7.5, Time-frequency check on PC 1". Show Figure 9.

One last check before the synthesis. PCA gave us the *spatial* mode
of ENSO — the horseshoe pattern. LIM gave us a global ``T = 3.1``
year, ``\tau_d = 8`` month oscillator. The piece in between is
checking that PC-1 *as a time series* actually has the right local
frequency content, not just the right average period. That's what a
wavelet spectrogram answers.

I ran a Morlet continuous wavelet transform on PC-1, following
Torrence and Compo, 1998. This is **Lectures 15 and 27 in action**,
the wavelet basis from Lecture 15 and the time-frequency
representation from Lecture 27. Implementation from scratch via the
FFT.

Power is concentrated in the two-to-seven year band, *exactly the
canonical ENSO range*. The strongest excursion is the 1997-98
super-El-Niño. There are quieter epochs in the 1960s and 70s and a
more active period from the 1980s on, a known multidecadal
modulation.

The takeaway: the recovered ENSO mode has the right temporal as well
as the right spatial structure.

---

## 9:55 to 10:30, Section 8, Summary, and Works cited

▸ Scroll to "## 8, Summary". The pass/fail table is visible.

To pull the four threads together. Going back to the climate-physics
claim at the start: *the tropical Pacific climate state lives on a
low-dimensional attractor, and ENSO is its dominant mode.* That
claim is now verified along **four orthogonal directions**, none of
which used Niño 3.4 as input.

* **PCA**, the **Lecture 17** covariance decomposition, recovers the
canonical *spatial* horseshoe pattern at correlation 0.946.
* **Anisotropic diffusion maps**, the out-of-course method, recovers
the same axis at 0.923 *and* reveals the **curved geometry** of the
attractor that PCA cannot see, including the El Niño / La Niña
asymmetry.
* **The Linear Inverse Model**, the **Lecture 11** estimation
problem, recovers the **dynamical signature**: a damped oscillator
with period 3.1 years and e-folding decay 8 months.
* **The Morlet wavelet spectrogram**, **Lectures 15 and 27**,
confirms the **temporal structure**: power concentrated in the 2–7
year ENSO band.

Five of seven predictions pass. Two fail honestly: Takens-style
delay embedding does *not* improve the DMAP recovery, and the
preregistered prediction lets us report that as a real result instead
of quietly dropping it.

The deeper takeaway, the reason this is in the same course as
neural-manifolds work: **the same operator-theoretic toolkit
recovers low-dimensional structure in high-dimensional climate data
and in high-dimensional neural recordings**. PCA, diffusion maps,
LIM, the wavelet — these are not climate-specific tricks. They're
spectral decompositions of operators (covariance, transition matrix,
propagator, time-frequency representation) that any sufficiently
structured sampled dynamical system admits. The mathematical content
of "recovering an attractor" is shared between climate dynamics and
the computational neuroscience I'm interested in. **The substrate
doesn't change the math.**

▸ Scroll to "## 9, Works cited" at the bottom of the notebook to
acknowledge the references on screen for ~5 seconds: Coifman and
Lafon for diffusion maps, Penland and Sardeshmukh for LIM, Torrence
and Compo for the wavelet treatment, Jin's recharge oscillator,
Takens for the embedding theorem, plus Lorenz's 1956 EOF analysis,
Mante-Sussillo and Pandarinath for the neural-manifolds analogy.
Don't read each name aloud; point and say "the full reference list
is here".

Thank you. I'm happy to take questions.

---

# Anticipated questions, read aloud only if asked

## "Why is the LIM correlation only 0.61?"

Three reasons, none of them a problem.

One: LIM's natural representation of the ENSO state is two-dimensional,
not one-dimensional. The ENSO mode is a complex eigenpair; the in-phase
and quadrature components are 90 degrees apart in the oscillation cycle.
Niño 3.4 is a one-dimensional regional mean. Projecting onto either
axis alone captures only one component of the 2-D state.

Two: PCA is statistically optimal at correlating with linear functionals
of the data, and Niño 3.4 is roughly a linear functional of SST. LIM
is not optimised for that, it is optimised to recover the *propagator*.
Apples-to-oranges comparison.

Three: the point of LIM in this project is the dynamics axis, period
and damping rate, not the leading-mode correlation. Both are
recovered correctly: 3.1 years and 8 months, in the published literature
range.

## "Why don't the three alpha values give very different leading-mode correlations?"

Because ENSO is approximately *linear* at first order. The recharge
oscillator is a linear ODE in two variables. Linearised systems have
variance directions and manifold directions that coincide, so PCA
and any of the three alpha values all recover the same leading axis
at near-equal fidelity.

The differences appear in the *higher coordinates* and in the visible
curvature of the embedding. That's where the geometric content lives.

## "Could the time-delay extension be fixed with sparse delays or different embeddings?"

I tried sparse delays, tau equals three and tau equals six. That
was actually worse, not better. Distance concentration is more severe
for sparse-but-decorrelated lags than for dense-redundant ones. I
also tried delay-embedding the PC scores rather than the raw spatial
field. No improvement.

The conclusion: on monthly ENSO data, the static state-space
embedding suffices, and the delay refinement does not pay back its
dimensional cost.

## "Would diffusion maps win on a dataset where PCA does not work as well?"

Yes, likely. There's a hyperspectral dataset I explored before
pivoting, agricultural lettuce at four growth stages, where PCA's
PC-1 captured most of the brightness variation but failed to recover
the growth-stage ordering. Diffusion maps' Psi-two was strictly
monotone in growth week. Same mathematical lesson, different
substrate.

The rule is: PCA is enough when you only need a one-dimensional index.
Diffusion maps wins when you need a *coordinate system* on a
nontrivially-curved manifold.

## "Why square-root of cosine of latitude weighting, and not cosine?"

The square-root goes on the *data matrix*. The implicit area weight
on the *covariance matrix* is then cosine of latitude, which is the
right physical weight for spherical surface area. This is standard
practice in EOF analysis since Bretherton, Smith, and Wallace, 1992.

## "Is the cone of influence in the wavelet figure cutting off real signal?"

Above sixteen years, the cone of influence dominates and the period
band is unreliable. Below sixteen years, including the entire three-
to-seven year ENSO band, the cone is well below the band, so the
multidecadal modulation visible in the figure is real signal, not an
edge artefact.

---

# Extended Q&A bank

> Use the section headings below to find a question quickly.
> Each answer is written for read-aloud delivery, with key numbers and
> citations spelled out.

## A. Process and experience questions

### "How did you choose this project topic?"

I'm interested in dimensionality reduction and what's called neural
manifolds, low-dimensional structure inside high-dimensional neural
recordings. ENSO struck me as the climate analogue of that: a
low-dimensional dynamical system embedded in a very high-dimensional
state space, the SST anomaly field. The mathematical structure is
the same, the substrate is different. So the project is partly a
sandbox for the manifold-recovery techniques I want to use in my own
research.

### "Did you have a result you were hoping for?"

I expected PCA to recover ENSO with a high correlation. That's a
known result going back to Lorenz 1956. The genuinely open question
was what diffusion maps would add. My expectation was that
diffusion maps would either match PCA on the leading mode or
slightly exceed it, with the real difference appearing in the
geometry of higher coordinates. That is what happened.

### "What was the most surprising part of doing this project?"

Two things. First: the time-delay embedding test. By Takens'
theorem, stacking time-delayed copies of a state vector should give
a better attractor reconstruction. Empirically, on this dataset, it
makes things worse. The diagnosis is interesting. Distance
concentration in high-dimensional space plus eigenfunction reordering.
Second: how clean the LIM eigenpair came out at 3.12 years and 8
months. Those numbers are exactly in the published-LIM literature
range, and they fell out of a few lines of matrix algebra.

### "How long did the project take?"

About a week of focused work, maybe forty to fifty hours including
the rewriting and the figures. The first half was scaffolding and
data handling, the second half was the analysis and the writeup.

### "Did you write everything yourself?"

The algorithms, yes, all from scratch in Julia, except for two
primitives: the SVD function `LinearAlgebra.svd` and the
eigendecomposition `LinearAlgebra.eigen`. The matrix logarithm in
the LIM uses `Base.log` extended to matrices. Everything else, the
PCA wrapper, the anisotropic α-family kernel construction, the
Nyström extension, the Morlet wavelet, the LIM, is in `src/`,
about seven hundred lines of Julia.

## B. Technical detail questions

### "What does it actually mean to take the SVD here?"

The data matrix X is 900 months by 2300 spatial cells. The SVD writes
X as U times Sigma times V transpose. U has 900 rows, V has 2300
rows, and Sigma is diagonal with singular values along it. The
columns of V are spatial patterns, what climate scientists call
empirical orthogonal functions or EOFs. The columns of U times Sigma
are temporal patterns, the principal components. The truncation to
the first k columns is the optimal rank-k approximation of X in any
unitarily invariant norm, which is the Eckart-Young theorem.

### "What's the diffusion map kernel actually doing?"

It's building a graph. Each month is a node, and the edge weight
between months i and j is `exp(minus distance squared over epsilon)`,
a Gaussian similarity. After alpha-renormalization to remove
density bias, the matrix is row-normalized to a Markov transition
matrix. So you can think of it as a random walk on the graph. Two
months are close in the diffusion-map embedding if and only if the
random walk reaches them with similar probabilities. The leading
eigenvectors of the transition matrix are the slow modes of the walk;
they parameterize the manifold.

### "Why the symmetric similarity trick instead of diagonalizing P directly?"

P is row-stochastic, so it's not symmetric. Direct eigendecomposition
of a non-symmetric matrix is numerically poor. P happens to be
similar to a symmetric matrix, M equals D-to-the-minus-half times K
times D-to-the-minus-half, where K is the alpha-renormalized
similarity and D is the diagonal of row sums. The two have the same
spectrum. So I diagonalize M with the symmetric eigendecomposition,
which is fast and gives real eigenvalues, then undo the similarity to
get the right eigenvectors of P via psi-k equals D-to-the-minus-half
times v-k.

### "Why does alpha equals 1 specifically recover Laplace-Beltrami?"

There's an asymptotic calculation. As epsilon goes to zero, the
generator of the Markov chain converges to Laplace-Beltrami plus a
drift term that's proportional to one-minus-two-alpha times the
gradient of log density. At alpha equals one, the drift term
vanishes, and you're left with just the geometric Laplacian. That's
why alpha equals one is sampling-density-invariant, which is what we
want when the climate state is sampled unevenly in time.

### "How does LIM recover the ENSO period?"

Assume the state evolves as dx/dt equals L times x plus noise. The
lag-tau covariance satisfies C-of-tau equals exp of L-tau times
C-of-zero. Estimating C-of-tau and C-of-zero from data and inverting
gives the propagator. Take the matrix logarithm to get L. The
eigenvalues of L are complex; each eigenpair is a damped oscillator.
Period equals 2 pi over the imaginary part. Damping equals minus
one over the real part. The least-damped complex eigenpair gave me
period 3.12 years and damping 8 months for ENSO.

### "What's the Morlet wavelet, and why use it?"

It's a Gaussian-windowed complex exponential at non-dimensional
frequency 6, the standard choice in climate-science wavelet
applications since Torrence and Compo, 1998. The convolution with
the signal at scale s gives a complex coefficient whose magnitude
squared is the local power at period roughly 1.03 times s. Computed
via FFT. You see power, frequency, and time on the same plot.

## C. Decision questions

### "Why ERSST version 5 over HadISST?"

Three reasons. One: NOAA's Niño 3.4 index is computed from ERSST, so
correlating with it is direct. Two: ERSST goes back to 1854, giving
75 years of post-1950 coverage. HadISST also has good coverage but
at 1-degree resolution requires five times the storage. Three:
2-degree resolution is well above the Nyquist limit for ENSO, which
is basin-scale, so the higher resolution doesn't add information.

### "Why the tropical Pacific only? Why not global?"

ENSO is a tropical Pacific phenomenon spatially. Including the
global SST adds variance from unrelated processes, the North Atlantic
Oscillation, the Pacific Decadal Oscillation, the Indian Ocean
Dipole, that compete with the ENSO axis for leading-EOF status.
Restricting to the canonical ENSO domain tightens the leading-mode
correlation by about two to four percentage points and produces a
cleaner first-EOF spatial pattern. There's nothing special about the
tropical Pacific mathematically, it's just where the signal of
interest lives.

### "Why subtract monthly climatology rather than the annual mean?"

The seasonal cycle, summer warmer than winter, is several times
larger than the ENSO interannual signal at any single grid cell. If
I subtracted only the annual mean, PC 1 would be the seasonal cycle,
not ENSO. Subtracting the monthly climatology, the 1950 to 2024
mean for each calendar month separately, removes the seasonal mode
and leaves only interannual anomalies.

### "Why didn't you use Nyström extension for ENSO?"

The DMAP problem here is small. 900 monthly snapshots gives a
900-by-900 affinity matrix, six megabytes, fits in cache. I kept
the Nyström machinery in the diffusion module for completeness, but
it's not needed for this dataset. It would matter if I went to
daily data, around 27000 daily snapshots.

## D. Field-connection questions

### "Why is ENSO modeled as low-dimensional?"

The leading dynamical model is the recharge oscillator from Jin,
1997. Two coupled ODEs in two state variables, eastern Pacific SST
anomaly and western Pacific thermocline depth. That's two
dimensions. Adding the seasonal cycle and ENSO Modoki gives roughly
five intrinsic dimensions, which matches the spectral gap I see in
the DMAP eigenvalues.

### "How does this relate to neural manifolds?"

The mathematical setup is identical. You have high-dimensional
observations, the SST field or population firing rates, generated
by a smooth function of low-dimensional latent variables, the
climate state or task variables. You want to recover the
low-dimensional manifold and its geometry from the observations
alone. The same toolkit, factor analysis or PCA on the linear side,
diffusion maps or Isomap on the nonlinear side, gets used in both
fields. Coifman and Lafon's α-family kernel is in active use in
computational neuroscience for hippocampal place-cell manifolds and
motor-cortex preparatory subspace.

### "Could you forecast Niño 3.4 with these methods?"

PCA and DMAP, no. They are static state-space methods. They tell
you what the climate state is, not how it will evolve. LIM does
forecast: the propagator G of tau equals exp of L tau lets you push
the state forward in time. Penland and Sardeshmukh used LIM to do
seasonal forecasts of ENSO, and on average it does as well as
operational dynamical models out to about six months.

### "Where does diffusion maps actually fail?"

Where the manifold hypothesis fails. In particular: when the data
has multiple disconnected components, diffusion maps gives spurious
modes from the disconnection rather than the geometry. When the
data is too sparse to estimate the kernel locally, again it fails.
And when the noise level is comparable to the manifold curvature,
you recover the noise, not the manifold.

## E. Class content connections (questions a peer might ask about previously-covered topics)

### "In Lecture 17 we computed the empirical covariance and eigendecomposed it. You did SVD on the data matrix. Are those the same thing?"

Yes, they're the same up to a normalization. If X is the centered
T-by-N data matrix, the empirical covariance is C equals X-transpose
times X over N minus 1. Eigenvectors of C are columns of V, where
X equals U Sigma V-transpose, and eigenvalues of C are sigma-squared
over N minus 1. So SVD of X gives the same EOFs and same eigenvalues
the Lecture 17 procedure would; it's just numerically more stable
because we don't square the condition number by forming X-transpose
X explicitly.

### "Lecture 18's example was EOF analysis, exactly your PCA. So what's new in your project?"

Two things. The first method, PCA, is essentially the Lecture 18
result restricted to tropical-Pacific SST, included as the canonical
baseline and to confirm the textbook ENSO horseshoe. The new content
is the comparison: an out-of-course nonlinear method, anisotropic
diffusion maps, that recovers the *curved geometry* of the attractor
that PCA's linear projection cannot see; and a Linear Inverse Model
that recovers ENSO's *dynamics* that neither static method can give.
The PCA result is the foothold; the contribution is what the other
methods reveal beyond it.

### "The wavelet transform via FFT — that's the convolution theorem from Lectures 5 and 7?"

Exactly. The Morlet CWT at scale s is a convolution of PC 1 against
a dilated complex wavelet. Direct convolution is order T times S
(signal length times number of scales); via FFT it's T log T per
scale. The convolution theorem from Lectures 5 and 7 is what makes
the spectrogram tractable. The implementation in `src/spectrogram.jl`
is FFT-multiply-inverse-FFT.

### "Lecture 9 was on bandpass filtering. Why a wavelet instead of just bandpassing PC 1 in the 3-7 year band?"

A bandpass filter would tell us *how much* power is in the ENSO band
on average. The wavelet tells us how that power is *distributed in
time*: when ENSO was active, when it was quiet, whether the 1997-98
super-El-Niño shows up as a localized burst in the spectrogram. For
detecting non-stationarity in ENSO amplitude, which is exactly what
the multidecadal modulation in Figure 9 shows, the time-frequency
representation from Lecture 27 is the right tool, not a stationary
bandpass.

### "The Gaussian kernel in DMAP — same Gaussian as the smoothing kernel from Lecture 8?"

Same functional form, different role. Lecture 8 used Gaussians for
smoothing on a regular grid: convolve a signal with a fixed-width
Gaussian to attenuate high-frequency noise. In DMAP I use the
Gaussian as a *similarity kernel* between data points, exp of
minus-distance-squared over epsilon. The bandwidth epsilon is the
same conceptual knob, but here it sets the scale at which the random
walk treats two months as neighbors in the data graph, not the
spatial scale at which a smoothing filter rolls off.

### "Lecture 11 was Estimation. Walk me through LIM as the maximum-likelihood estimator."

Assume the state evolves as dx/dt equals L x plus Q xi with xi
vector white noise. Conditioned on x at time t, the state at t plus
tau is Gaussian with mean exp of L tau times x-of-t and a covariance
that doesn't depend on x. The negative log likelihood is quadratic
in L, so maximizing the likelihood reduces to least squares for the
propagator G of tau equals C of tau times C of zero inverse, then
L equals log G of tau over tau via the matrix logarithm. That's
the textbook ML estimation procedure from Lecture 11 applied to a
linear stochastic ODE.

### "Lecture 21 covered different similarity measures. Why Pearson correlation against Niño 3.4 specifically?"

For external validation, Pearson correlation against Niño 3.4 is the
literature-standard ENSO metric, so I used it for direct
comparability with published results. Internally, the DMAP kernel is
a *different* similarity, the Gaussian on Euclidean distance, which
is the local nonlinear version of Pearson similarity. The two
similarities serve different purposes: kernel for the embedding,
linear correlation for the comparability check.

### "Lecture 22 was Sparsity. Why not use sparse PCA for the EOFs?"

Sparse PCA would force most loadings of each EOF to be zero, which
gives spatial patterns that look like *localized* features instead
of the smooth basin-scale horseshoe. For ENSO that would probably
collapse EOF 1 onto the equatorial warm tongue and remove the
subtropical horseshoe lobes. I didn't use it because the unmodified
EOF horseshoe *is* the canonical ENSO reference pattern, and the
project's hypothesis was about recovering exactly that. Sparse PCA
would be the right tool if I were trying to *discover* compact
regional predictors instead of validating the textbook mode.

### "Lecture 24 was about distance metrics. Why Euclidean distance for the DMAP kernel?"

The Euclidean distance on the area-weighted SST anomaly vector
corresponds to the L-2 norm on the SST field on the sphere, which
is the natural physical metric for "how different are these two
months". Other choices, cosine distance for example, would lose the
amplitude information that distinguishes a strong El Niño from a
weak one. The square-root cosine of latitude weighting on each
column is what makes Euclidean on the data matrix equivalent to
area-weighted L-2 on the sphere, see Bretherton, Smith, and Wallace,
1992.

### "Lecture 28 was Adaptive Regularization. Could the DMAP bandwidth be made adaptive?"

Yes, and there's a literature on it. Self-tuning diffusion maps
(Zelnik-Manor and Perona, 2004) uses a per-point epsilon equal to
the distance to the k-th nearest neighbor of each point, which adapts
to local density variations. I used a global epsilon equal to the
median squared pairwise distance, but the bandwidth-sensitivity
figure (Figure 8) shows the leading-mode correlation plateaus across
a factor of about thirty in epsilon, so the result isn't bandwidth-
limited and adaptive bandwidth wouldn't change the answer. Adaptive
bandwidth would matter on data with more extreme density variation
than monthly ENSO has.

### "Lecture 12 was Conjugate Gradient. Anywhere CG could have applied?"

Not in this project. My pipeline is purely spectral: SVD for PCA,
symmetric eigendecomposition for DMAP, matrix logarithm for LIM,
no iterative linear solve. CG would matter if I were solving an
inverse problem, for instance reconstructing a dense SST field from
sparse satellite or buoy observations, or doing 4D-VAR data
assimilation. None of that is in scope here.

### "Lecture 20 was Streaming. Could LIM be made streaming for online ENSO forecasting?"

In principle, yes. The LIM propagator G of tau can be updated
recursively as new monthly data arrives, by a Kalman-style update on
the lag-tau covariance. Penland and Sardeshmukh's 1995 paper actually
did near-real-time LIM forecasts of ENSO using exactly that kind of
incremental update. I didn't implement it because the project's
question is whether the static fit recovers the correct period and
damping rate, not whether we can forecast ENSO online. But the
extension is straightforward.

### "Are there other topics we covered that you didn't use, and would they have helped?"

Triangulation (Lecture 14), Shaping (16). Triangulation comes up for
irregularly-sampled data; ERSSTv5 is on a regular 2-degree grid, so
it's not relevant. Shaping is for inverse problems with structured
priors; my pipeline is forward spectral analysis, no inverse problem.
The ones that would actually have *added* something on this dataset
are Sparsity (22) for interpretable EOFs, and Adaptive Regularization
(28) for self-tuned bandwidth. Both are reasonable extensions, and
both would address marginal questions rather than the core
attractor-recovery hypothesis.

## F. Critical questions / pushbacks

### "PCA recovers ENSO at 0.95, DMAP at 0.92. Isn't PCA strictly better?"

If your goal is a 1-D scalar index of ENSO state, yes, PCA is fine
and simpler. PCA wins at correlating with linear functionals like
Niño 3.4. But that's not what diffusion maps is for. Diffusion maps
gives you a *coordinate system* on the attractor, including the
curvature, the El Niño / La Niña asymmetry, and the trajectory
through state space. PCA gives you a featureless cloud in
two-dimensional space; DMAP gives you a clean horseshoe. The
methods answer different questions.

### "The time-delay embedding fails. Doesn't that suggest the project's core idea is wrong?"

No, it suggests that *naive* method-stacking is not free. The core
result, that PCA and DMAP both recover ENSO, doesn't depend on
delay embedding. The time-delay test was a robustness check on top
of the working method, asking whether stacking another technique
helps. It doesn't, on this dataset, for diagnosable reasons. A
careful reader should *like* that the negative result is reported
and diagnosed, not buried.

### "Couldn't I argue you're just rediscovering known results?"

Yes, that's exactly what unsupervised recovery means. ENSO is
well-known. The point of the project is showing that three
independent methods, given only the SST anomaly field and no labels,
recover the spatial pattern, the dynamical period, and the curved
geometry that the climate-physics literature predicts. The methods
don't *know* about ENSO; they find it. The fact that they all
converge on the same answer is the content.

### "Why didn't you use a deep-learning method like an autoencoder?"

Autoencoders are nonlinear dimensionality reducers and would
plausibly work. I didn't use one because I wanted methods with
clean operator-theoretic foundations, not black-box function
approximators. A deep autoencoder gives you a low-dimensional
embedding, but you don't get a kernel, a transition operator, or
eigenvalues with explicit physical meaning. PCA gives you variance,
DMAP gives you Laplace-Beltrami eigenfunctions, LIM gives you a
period. Each piece of output has direct climate-physics
interpretation.

### "Could the result be different if you used more PCs in the LIM?"

Yes, slightly. I tried 5, 7, 10, and 15 PCs. The ENSO eigenpair has
period 3.1 to 6.2 years across that range, with damping consistently
around 6 to 8 months. The period at 7 PCs is on the short end at
2.9 years, at 5 PCs on the longer end at 6.2 years. I chose 10 PCs
as a standard Penland-Sardeshmukh number; the result at that choice
is 3.12 years and 0.68 years, which is in the middle of the
published range.
