### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6910d5fb-0522-499b-8210-d09950db41ad
begin
	# Listed here so Pluto's auto-Pkg sees every dependency.
	using LinearAlgebra
	using Statistics
	using StatsBase
	using Random
	using Printf
	using Dates
	using PlutoUI
	using CairoMakie
	using ColorSchemes
	using NCDatasets
	using Distances       # required by src/diffusion.jl
	using NearestNeighbors   # required by src/metrics.jl (utility)
	using FFTW         # required by src/spectrogram.jl
	using Logging
	# silence k-means convergence warnings if anything pulls Clustering
	global_logger(SimpleLogger(stderr, Logging.Error))
end

# ╔═╡ 30d4a0b5-4ab0-45fa-b7c3-adc3a15c2edb
begin
	const SRC = joinpath(@__DIR__, "..", "src")
	include(joinpath(SRC, "load_sst.jl"))
	include(joinpath(SRC, "pca.jl"))
	include(joinpath(SRC, "diffusion.jl"))
	include(joinpath(SRC, "delay_embed.jl"))
	"loaded"
end

# ╔═╡ d03ddc18-c35c-4fd8-9f92-e499c1c345c4
md"""
# Recovering ENSO-related structure from sea-surface temperature

**Unsupervised representation learning on monthly tropical-Pacific SST**

GEO 384H project · William Suan

Can a few learned coordinates summarize variation associated with El Niño–Southern Oscillation (ENSO), and how does that representation change when temporal context is included?

The computational work combines SST preprocessing, PCA/EOF analysis, anisotropic diffusion maps, parameter sensitivity and control experiments, and delay embeddings in both raw-field and PCA feature spaces. A Linear Inverse Model (LIM) extends the comparison from snapshot geometry to lagged dynamics. The Julia modules implement these established methods using SVD, symmetric eigendecomposition, and a matrix logarithm.

The saved analysis reports same-month Niño 3.4 correlations of **0.946 for PCA** and **0.923 for diffusion maps at α = 1**. Adding 12 monthly lags lowers the leading diffusion-coordinate correlation to **0.470** in raw-field space. These results make the linear baseline and the unsuccessful temporal-feature experiment equally important to the comparison.

Preprocessing and fitting use the full 1950–2024 record. Niño 3.4 supplies a physically interpretable SST reference for sign alignment, coloring, and descriptive correlations. Its shared SST basis and the temporal dependence of monthly observations define the scope of this evaluation. Numerical values labeled “saved” below come from the repository's metric files; interpolated values come from the notebook's active computation.
"""

# ╔═╡ a51c45f4-1f89-45e8-9967-b5b4c3bc6835
md"""
## 0, Setup

Load the Julia libraries and project modules for preprocessing, spectral representations, delay features, and temporal diagnostics. PCA uses SVD; diffusion maps uses a symmetric eigendecomposition of the normalized affinity matrix.
"""

# ╔═╡ 545ecb96-47c5-44f2-82ed-178441e4b0ee
md"""
## 1, Data and feature construction

NOAA ERSSTv5 provides monthly SST on a 2° grid. The analysis selects January 1950 through December 2024 and the tropical Pacific, 30°S–30°N and 120°E–80°W. The recorded matrix has 900 monthly observations and 2,322 retained grid-cell features. Restricting the domain focuses the variance and similarity calculations on the basin associated with ENSO.

For grid cell ``j``, subtract its calendar-month climatology estimated over the analysis window:
```math
a_{tj}=s_{tj}-\overline{s}_{m(t),j}.
```
This removes the mean seasonal cycle while retaining departures from the usual temperature for that month. `flatten_to_matrix` reshapes the cube into a month-by-cell matrix and excludes any column containing a NaN anywhere in the selected record. The features retain their relative temperature amplitudes; there is no per-cell variance standardization or detrending in this pipeline.

The intended area weighting is
```math
x_{tj}=a_{tj}\sqrt{\cos\phi_j},\qquad
\|x_t-x_u\|^2=\sum_j\cos\phi_j\,(a_{tj}-a_{uj})^2.
```
With weights matched to columns, this accounts for the latitude-dependent area of a regular angular grid. **Implementation limitation:** `flatten_to_matrix` traverses spatial cells in Julia's latitude-fast column order, while `cos_lat_weights` appends weights longitude-fast. The saved results therefore use a weight ordering that needs correction and re-evaluation before claiming this exact area-weighted metric.

Niño 3.4 averages SST anomalies over 5°S–5°N, 170°W–120°W. The code matches it to SST dates and excludes missing reference values from correlations. Its role is interpretation after embedding: eigenvector signs are aligned with it and plots are colored by it. Since the index and features both describe SST, agreement measures recovery of a familiar regional signal within the fitted record.
"""

# ╔═╡ 94c1b1bd-6bbb-447d-963c-7d37569209b3
begin
	# Step 1: download the ERSSTv5 netCDF file from NOAA PSL if not cached.
	if !isfile(SST_PATH)
		println("Downloading ERSSTv5 (~150 MB)...")
		download("https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc", SST_PATH)
	end
	# Step 2: load it into a (T × lat × lon) Float32 cube; missing → NaN.
	sst, lat_all, lon_all, time_all = load_ersst()
	# Step 3: trim to the 1950–2024 analysis window.
	keep = findall(t -> Date(1950,1,1) <= Date(t) <= Date(2024,12,31), time_all)
	sst_w = sst[keep, :, :]
	time_w = time_all[keep]
	(; T = size(sst_w,1), lat = size(sst_w,2), lon = size(sst_w,3),
	  first = Date(time_w[1]), last = Date(time_w[end]))
end

# ╔═╡ d5c1a5ac-248a-439c-a45a-56a053a1cfad
begin
	# Step 1: subtract the monthly climatology, the 1950–2024 mean for each
	# calendar month at each grid cell. This removes the seasonal cycle (which
	# is several times larger than the ENSO interannual signal) and isolates
	# anomalies relative to "what's normal for this month at this location."
	anom = sst_anomalies(sst_w, time_w)
	# Step 2: restrict to the tropical Pacific.
	pac, lat, lon = subset_region(anom, lat_all, lon_all)
	# Step 3: reshape (T × lat × lon) → (T × N) matrix and drop land cells.
	X_raw, mask, _ = flatten_to_matrix(pac)
	# Step 4: apply √cos(latitude) area weighting. Earth is a sphere, so each
	# grid cell at latitude φ represents physical area ∝ cos(φ). Multiplying
	# the data matrix columns by √cos(φ) makes Euclidean inner products on
	# rows equivalent to area-weighted physical inner products on the sphere.
	weights = cos_lat_weights(lat, lon, mask)
	X = X_raw .* reshape(weights, 1, :)
	(; X_size = size(X), N_sea = size(X,2), land_frac = round(1 - size(X,2) / length(mask); digits=3))
end

# ╔═╡ 6c8313b3-08ca-4a7a-b130-3678b9dfa234
begin
	if !isfile(NINO34_PATH)
		download_nino34()
	end
	n_t, n_v = load_nino34()
	ix = [findfirst(==(Date(t)), n_t) for t in time_w]
	nino34 = Float64[isnothing(j) ? NaN : n_v[j] for j in ix]
	valid_n = .!isnan.(nino34)
	(; n_months_valid = count(valid_n))
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000001
md"""
### Sample months: SST anomaly fields

Three selected anomaly fields show the spatial observations before feature weighting. The dashed box marks Niño 3.4, connecting the basin-wide input to the regional reference.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000002
let
	fig = Figure(size = (1500, 380))
	months_to_show = [(Date(1997, 12, 1), "Dec 1997, super-El-Niño"),
					   (Date(1989, 1, 1), "Jan 1989, selected month"),
					   (Date(1999, 12, 1), "Dec 1999, strong La Niña")]
	vmax = 4.0

	local sct
	for (j, (d, label)) in enumerate(months_to_show)
		idx = findfirst(==(d), Date.(time_w))
		ax = Axis(fig[1, j];
				  title = label,
				  xlabel = "longitude (°E)",
				  ylabel = j == 1 ? "latitude (°N)" : "",
				  aspect = 2.5)
		slice = pac[idx, :, :]
		sct = heatmap!(ax, lon, lat, slice';
					   colormap = :RdBu_11, colorrange = (-vmax, vmax))
		poly!(ax, Point2f[(190, -5), (240, -5), (240, 5), (190, 5)];
			  color = (:black, 0), strokecolor = :black, strokewidth = 2,
			  linestyle = :dash)
	end
	Colorbar(fig[1, 4], sct; label = "SST anomaly (°C)")
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000003
md"""
### Hovmöller diagram: equatorial anomalies through time

This heat map averages anomalies over 2°S–2°N and shows longitude against time. It gives context for the monthly fields being summarized. Read warm and cool signs from the colorbar.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000004
let
	eq_idx = findall(l -> -2 <= l <= 2, lat)
	Lo = length(lon)
	Tn = length(time_w)
	eq_strip = fill(NaN32, Tn, Lo)
	for t in 1:Tn, i in 1:Lo
		vals = filter(!isnan, [pac[t, j, i] for j in eq_idx])
		if !isempty(vals)
			eq_strip[t, i] = mean(vals)
		end
	end
	years_axis = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1)/12 for t in time_w]

	fig = Figure(size = (1100, 720))
	ax = Axis(fig[1, 1];
			  title = "Equatorial Pacific SST anomaly (2°S–2°N average), longitude × time",
			  xlabel = "longitude (°E)", ylabel = "year")
	hm = heatmap!(ax, lon, years_axis, eq_strip';
				  colormap = :RdBu_11, colorrange = (-3.5, 3.5))
	Colorbar(fig[1, 2], hm; label = "SST anomaly (°C)")

	# Label a small set of marquee events on the heat map. Years that sit
	# close together (1997-98 vs 1998-99, 1982-83 vs the surrounding years)
	# are skipped to keep the labels readable.
	for (yr, label) in [(1972.7, "1972 El Niño"),
						(1982.7, "1982-83 El Niño"),
						(1988.7, "1988 La Niña"),
						(1997.7, "1997-98 El Niño"),
						(2015.7, "2015-16 El Niño"),
						(2023.7, "2023 El Niño")]
		text!(ax, 130, yr; text = label, color = :white,
			  fontsize = 11, font = :bold,
			  strokecolor = :black, strokewidth = 1.5)
	end
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000005
md"""
### Niño 3.4 reference series

The fill distinguishes positive and negative anomalies. The dashed ±0.5 °C lines are guides, not an event-classification rule for individual months. Operational classifications also depend on averaging, persistence, and other conditions.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000006
let
	years_axis = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1)/12 for t in time_w]
	fig = Figure(size = (1300, 360))
	ax = Axis(fig[1, 1];
			  title = "Niño 3.4 index, banded by ENSO phase",
			  xlabel = "year", ylabel = "Niño 3.4 anomaly (°C)")

	zero_line = fill(0.0, length(years_axis))
	band!(ax, years_axis, zero_line, max.(nino34, 0.0); color = (:crimson, 0.55))
	band!(ax, years_axis, min.(nino34, 0.0), zero_line; color = (:steelblue, 0.55))
	lines!(ax, years_axis, nino34; color = :black, linewidth = 0.5)
	hlines!(ax, [0]; color = :gray, linewidth = 0.5)
	hlines!(ax, [0.5, -0.5]; color = :gray, linestyle = :dash, linewidth = 0.7)
	text!(ax, 1951.5, 0.55; text = "Reference level (+0.5°C)",
		  fontsize = 9, color = :gray)
	text!(ax, 1951.5, -0.85; text = "Reference level (−0.5°C)",
		  fontsize = 9, color = :gray)

	for (d, label) in [(Date(1972,12,1), "'72"), (Date(1982,12,1), "'82"),
					   (Date(1997,12,1), "'97"), (Date(2015,12,1), "'15"),
					   (Date(2023,12,1), "'23")]
		idx = findfirst(==(d), Date.(time_w))
		if !isnothing(idx)
			text!(ax, years_axis[idx], nino34[idx] + 0.15; text = label,
				  fontsize = 11, color = :black, font = :bold,
				  align = (:center, :bottom))
		end
	end
	fig
end

# ╔═╡ a2fed5c7-0243-4eea-b187-7f24d7feb845
md"""
## 2, Questions for the comparison

1. How much SST variance does PCA compress, and how closely does PC1 track Niño 3.4?
2. How do diffusion-map coordinates change with density normalization, bandwidth, and diffusion time?
3. Does adding temporal context through raw-field or PCA-space delay features improve index recovery?
4. What do label permutations, spatial shuffling, and lagged-covariance modeling reveal about the representations?

The historical hypotheses in `report/plan.md` include leading-coordinate correlation targets and a predicted improvement of at least 0.05 with 12 lags. Here those targets organize descriptive comparisons. Representation quality beyond index agreement, physical interpretation of higher coordinates, and performance on new months require additional evaluation.
"""

# ╔═╡ eaff9fa5-3072-4f2d-848b-6d6705f6b65e
md"""
## 3, PCA: the linear baseline

Each month is a point in a high-dimensional feature space. PCA finds an orthogonal linear subspace that captures as much sample variance as possible. An **empirical orthogonal function (EOF)** is a spatial loading pattern; its **principal component (PC)** gives the monthly amplitude of that pattern.

`fit_pca` centers each feature and computes
```math
X_c=X-\mathbf{1}\mu^\top=U\Sigma V^\top,\qquad
Z=X_cV=U\Sigma.
```
The columns of ``V`` are EOFs in weighted feature space. The covariance is ``C=X_c^\top X_c/(T-1)``, with eigenvalues ``\sigma_j^2/(T-1)``. The retained variance fraction is
```math
r_j=\frac{\sigma_j^2}{\sum_\ell\sigma_\ell^2}.
```
The denominator uses the full singular spectrum even when only 20 modes are retained. Truncating to ``k`` modes gives ``X_{c,k}=U_k\Sigma_kV_k^\top``, the best rank-``k`` reconstruction in Frobenius norm. Adding the fitted mean returns to the original feature coordinates.

This gives a concrete compression baseline for the nonlinear method. Niño 3.4 correlation is an empirical interpretation metric, distinct from PCA's variance objective. Flipping a matched EOF/PC pair changes its sign convention while preserving the reconstruction and absolute correlation. `project_pca` also implements projection with a fitted mean and basis, ``(Y-\mathbf{1}\mu^\top)V``, for new feature rows.
"""

# ╔═╡ 1e3a6b29-61f7-46f4-a2a8-32f449faaf27
begin
	# Compute the SVD of the centred (T × N) data matrix, truncated to the top
	# 20 modes. `fit_pca` returns the principal-component scores (U·Σ), the
	# right singular vectors V (= the EOFs, spatial patterns), and the
	# explained-variance fractions σᵢ²/Σσⱼ².
	pca = fit_pca(X; rank=20)
	PC = Float64.(pca.scores)  # (T × 20), temporal patterns
	EOF = Float64.(pca.V)    # (N × 20), spatial patterns

	# Eigenvectors are determined only up to sign. To make the leading PCs
	# positively correlated with Niño 3.4 (which is just a labeling convention,
	# not data leakage, Niño 3.4 enters only via the sign), we flip whichever
	# eigenpairs come out negative.
	for k in 1:5
		if cor(PC[valid_n, k], nino34[valid_n]) < 0
			PC[:, k] .= -PC[:, k]
			EOF[:, k] .= -EOF[:, k]
		end
	end

	# Correlations of each leading PC against the validation index.
	ρ_PC = [cor(PC[valid_n, k], nino34[valid_n]) for k in 1:5]
	(; ρ_PC1 = round(ρ_PC[1]; digits=3),
	  explained_top5 = round.(Float64.(pca.explained[1:5]); digits=3),
	  cumvar_top5 = round(sum(pca.explained[1:5]); digits=3))
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-400000000002
begin
	include(joinpath(SRC, "lim.jl"))
	# Standard Penland–Sardeshmukh: top 10 PCs, lag τ = 1 month
	lim_n_pcs = 10
	lim_τ   = 1
	lim    = fit_lim(PC[:, 1:lim_n_pcs]; τ = float(lim_τ))

	# Find ENSO mode (complex pair, period 2-8 yr, least damped, Im σ > 0)
	lim_periods = [imag(s) == 0 ? Inf : 2π / abs(imag(s)) / 12.0 for s in lim.σ]
	lim_decays = [real(s) == 0 ? Inf : -1.0 / real(s) / 12.0  for s in lim.σ]
	lim_cands  = findall(i -> 2.0 <= lim_periods[i] <= 8.0 &&
	              imag(lim.σ[i]) > 0,
	            eachindex(lim.σ))
	lim_idx   = lim_cands[argmin([abs(real(lim.σ[c])) for c in lim_cands])]

	# Project onto the mode → complex time series; align sign with Niño 3.4
	z_lim = project_to_mode(lim, PC[:, 1:lim_n_pcs], lim_idx)
	z_re = real.(z_lim)
	if cor(z_re[valid_n], nino34[valid_n]) < 0
		z_re = -z_re
		z_lim = -z_lim
	end
	ρ_LIM = cor(z_re[valid_n], nino34[valid_n])

	(; period_yr = round(lim_periods[lim_idx]; digits=2),
	  efold_yr = round(lim_decays[lim_idx]; digits=2),
	  ρ_Niño34 = round(ρ_LIM; digits=3))
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-200000000002
begin
	include(joinpath(SRC, "spectrogram.jl"))
	dt_yr = 1/12
	W_pc, scales_pc, periods_pc, coi_pc = morlet_cwt(PC[:, 1], dt_yr;
	                         dj = 1/16, s0 = 2*dt_yr)
	P_pc = abs2.(W_pc)
	"wavelet computed"
end

# ╔═╡ 848b1f1c-5edd-47e0-a1a3-a286c8dda937
md"""
**PC1 versus Niño 3.4:** $(round(ρ_PC[1]; digits=3)) in the active computation.

The saved correlation is 0.946, and the saved PC1 variance fraction is 0.401 (`figures/enso_metrics.txt`). Variance explained measures compression of the SST matrix; correlation measures agreement with the same-month regional reference.
"""

# ╔═╡ ce8f4301-cd81-4941-a57a-696c1b2806c3
md"""
### EOF 1, the leading spatial mode
"""

# ╔═╡ cc5ea7d9-b7cc-41a9-b73d-08e76a58c99b
function eof_grid(eof_vec, mask, weights)
	inv_w = 1.0 ./ weights
	eof_unw = eof_vec .* inv_w
	La, Lo = size(mask)
	G = fill(NaN32, La, Lo)
	flat_idx = findall(mask)
	for (k, ci) in enumerate(flat_idx)
		G[ci] = eof_unw[k]
	end
	return G
end

# ╔═╡ 0087a2a7-d84e-4252-9a72-be059d26099e
let
	G = eof_grid(EOF[:, 1], mask, weights)
	vmax = maximum(abs, filter(!isnan, G))

	fig = Figure(size = (1100, 460))
	ax = Axis(fig[1, 1];
	  title = "EOF 1, leading mode of tropical Pacific SST anomalies (1950–2024)",
	  xlabel = "longitude (°E)", ylabel = "latitude (°N)",
	  aspect = 2.5)
	hm = heatmap!(ax, lon, lat, G';
	  colormap = :RdBu_11, colorrange = (-vmax, vmax))
	Colorbar(fig[1, 2], hm; label="EOF 1 amplitude (a.u.)")

	# Niño 3.4 box overlay
	poly!(ax, Point2f[(190, -5), (240, -5), (240, 5), (190, 5)];
	   color = (:black, 0), strokecolor = :black, strokewidth = 2,
	   linestyle = :dash)
	text!(ax, 195, 6; text = "Niño 3.4 box", fontsize = 11)
	fig
end

# ╔═╡ f6088bba-972b-4e10-9a3e-df140b6a768d
md"""
The leading EOF maps PC1 loadings back onto the retained geographic cells after dividing by the applied weights. The dashed box locates Niño 3.4 within the broader spatial pattern. The colorbar defines loading signs; geographic interpretation also depends on correcting the weight-order issue described in Section 1.
"""

# ╔═╡ f512c08c-b1f4-4104-824f-f5e0c2506f2f
md"""
### PC 1 vs the Niño 3.4 index
"""

# ╔═╡ 52acae2e-4dfc-4394-9404-546addc5a228
let
	years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]
	s = std(nino34[valid_n]) / std(PC[:, 1])
	pc_scaled = PC[:, 1] .* s

	fig = Figure(size = (1300, 360))
	ax = Axis(fig[1,1];
	  title = "PC 1 vs Niño 3.4  (ρ = $(round(ρ_PC[1]; digits=3)))",
	  xlabel = "year", ylabel = "anomaly (Niño 3.4: °C)")
	lines!(ax, years, nino34;  color = (:steelblue, 0.85), label = "Niño 3.4 (NOAA)", linewidth = 1.5)
	lines!(ax, years, pc_scaled; color = (:crimson, 0.85),  label = "PC 1 (rescaled)", linewidth = 1.2)
	hlines!(ax, [0]; color = :gray, linewidth = 0.6)
	for d in [Date(1972,12,1), Date(1982,12,1), Date(1997,12,1),
	     Date(2015,12,1), Date(2023,12,1)]
		idx = findfirst(==(d), Date.(time_w))
		if !isnothing(idx)
			scatter!(ax, [years[idx]], [nino34[idx]];
			    color = :black, markersize = 10, marker = :star5)
		end
	end
	axislegend(ax; position = :rt, framevisible = false)
	fig
end

# ╔═╡ 147cf9a2-7582-48d1-bd4d-194b9d184480
md"""
PC1 is rescaled to the reference standard deviation for the overlay. The saved correlation of 0.946 indicates closely related same-month variation in these two SST summaries.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000007
md"""
### Higher-order EOFs and the variance budget

EOFs 2–4 describe orthogonal directions beyond the dominant pattern. The saved first-five variance fractions are 0.401, 0.155, 0.082, 0.043, and 0.038. The maps and variance fractions show what additional coordinates retain; assigning named physical processes to them requires further evidence.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000008
let
	fig = Figure(size = (1500, 700))
	for k in 1:4
		row = (k <= 2) ? 1 : 2
		col = ((k - 1) % 2) + 1
		G = eof_grid(EOF[:, k], mask, weights)
		vmax = maximum(abs, filter(!isnan, G))
		ax = Axis(fig[row, col];
				  title = "EOF $k  ($(round(100*pca.explained[k]; digits=1))% var)",
				  xlabel = row == 2 ? "longitude (°E)" : "",
				  ylabel = col == 1 ? "latitude (°N)" : "",
				  aspect = 2.5)
		heatmap!(ax, lon, lat, G';
				 colormap = :RdBu_11, colorrange = (-vmax, vmax))
		poly!(ax, Point2f[(190, -5), (240, -5), (240, 5), (190, 5)];
			  color = (:black, 0), strokecolor = :black, strokewidth = 1.2,
			  linestyle = :dash)
	end
	fig
end

# ╔═╡ 92eb9810-102f-4dbd-b618-604121d19f3b
md"""
## 4, Diffusion maps: nonlinear spectral features

Diffusion maps builds a weighted graph of monthly fields. Similar fields have stronger connections, and the leading nonconstant eigenfunctions describe slowly varying patterns on that graph. This offers a nonlinear alternative to PCA's linear variance directions. Graph-walk steps describe similarity propagation; physical time enters later through trajectories, delay features, and LIM.

For monthly feature vectors ``x_i``, `fit_diffusion_map` constructs
```math
K_{ij}=\exp(-\|x_i-x_j\|^2/\varepsilon),\qquad q_i=\sum_jK_{ij},
```
then adjusts sampling-density influence and normalizes rows:
```math
K^{(\alpha)}_{ij}=\frac{K_{ij}}{q_i^\alpha q_j^\alpha},\qquad
d_i=\sum_jK^{(\alpha)}_{ij},\qquad P=D^{-1}K^{(\alpha)},\quad D=\operatorname{diag}(d_i).
```
The default ``\varepsilon`` is the median squared distance over distinct pairs. Small bandwidths emphasize local neighbors; large bandwidths make affinities more similar across the sample. The implementation uses a dense all-pairs kernel including self-affinities.

For a numerically convenient real-symmetric eigenproblem, it diagonalizes
```math
M=D^{-1/2}K^{(\alpha)}D^{-1/2},\qquad Mv_j=\lambda_jv_j,
\qquad \psi_j=D^{-1/2}v_j.
```
Since ``P=D^{-1/2}MD^{1/2}``, these ``\psi_j`` are right eigenvectors of ``P``. The constant mode has eigenvalue 1 and is omitted from the embedding:
```math
\Phi_t(x_i)=(\lambda_2^t\psi_2(i),\ldots,\lambda_{d+1}^t\psi_{d+1}(i)).
```
The plots label these scaled coordinates Ψ₂, Ψ₃, and so on. Truncation retains the leading graph modes; their eigenvalue weights emphasize similarities that persist over multiple graph-walk steps.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-100000000001
md"""
### Density normalization and the α comparison

The ``q_i^\alpha q_j^\alpha`` factor reduces the influence of densely sampled neighborhoods. Under smooth-manifold, positive-density, and appropriate small-bandwidth/large-sample assumptions, the backward generator has the form, up to a positive kernel-dependent time scale,
```math
\mathcal L_\alpha f\;\propto\;\Delta_{\mathcal M}f+
2(1-\alpha)\,\nabla\log p\cdot\nabla f.
```
Here ``p`` is sampling density. Thus α = 0 retains density drift, α = 1/2 gives the density-weighted backward diffusion operator associated with a Fokker–Planck description, and α = 1 cancels the density-drift term to give a Laplace–Beltrami limit (Coifman & Lafon, 2006).

The α = 1 choice targets geometry with reduced sampling-density influence, useful when some SST configurations occur much more often than others. The finite-sample comparison tests sensitivity to that choice. Its theoretical motivation concerns the graph construction; establishing a physical climate attractor would require separate dynamical evidence.
"""

# ╔═╡ 369b550f-b097-4470-ae92-e601394f5db0
@bind α_choice PlutoUI.Select([0.0 => "α = 0 (graph Laplacian)",
               0.5 => "α = 1/2 (Fokker–Planck)",
               1.0 => "α = 1 (Laplace–Beltrami)"]; default=1.0)

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-300000000001
md"""
**Diffusion time** ``t`` controls spectral weighting through ``\lambda_j^t\psi_j``. Increasing ``t`` suppresses smaller-eigenvalue modes relative to the leading ones. The slider explores this change in relative coordinate scales for ``t=1,\ldots,5``. Scaling an individual nonzero coordinate by a positive constant leaves its Pearson correlation unchanged, so this is a geometric sensitivity view rather than a new correlation improvement.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-300000000002
@bind diff_t PlutoUI.Slider(1:5, default=1, show_value=true)

# ╔═╡ 992df9c2-7f00-433d-a4c8-26d02210ea6b
md"""
*Selected α = $α_choice, t = $diff_t*
"""

# ╔═╡ 90f298e4-6d17-4a81-b83c-169da804ff3b
begin
	# Fit the diffusion map: build the Gaussian kernel with bandwidth ε set
	# to the median squared pairwise distance (a robust default), α-renormalise
	# to remove sampling-density bias, row-normalise to a Markov transition
	# matrix P, then diagonalise the symmetric similarity M = D^(-1/2) K^(α) D^(-1/2)
	# (numerically stabler than diagonalising P directly) and undo the similarity
	# to recover right eigenvectors of P. Diffusion-time t powers eigenvalues
	# to amplify the gap between slow and fast modes.
	dm = fit_diffusion_map(X; α=α_choice, d=15, t=diff_t)
	Ψ = Float64.(dm.coords)

	# Sign-align each leading coordinate with Niño 3.4 (cosmetic, eigenvector
	# signs are arbitrary).
	for k in 1:5
		if cor(Ψ[valid_n, k], nino34[valid_n]) < 0
			Ψ[:, k] .= -Ψ[:, k]
		end
	end
	ρ_Ψ = [cor(Ψ[valid_n, k], nino34[valid_n]) for k in 1:5]
	(; ε = round(dm.ε; sigdigits=4),
	  ρ_Ψ1 = round(ρ_Ψ[1]; digits=3),
	  λ_top10 = round.(abs.(Float64.(dm.λ[1:10])); digits=4))
end

# ╔═╡ befa247c-0549-47d6-a30b-6a6fbad9ce3e
md"""
**Ψ₂ versus Niño 3.4:** $(round(ρ_Ψ[1]; digits=3)) at the selected settings.

The saved α = 1, t = 1 correlation is 0.923. This quantifies same-month index agreement within the full-sample embedding.
"""

# ╔═╡ 506a5a59-29a5-447c-b0f9-1afe70b75251
md"""
### Eigenvalue spectrum
"""

# ╔═╡ 880a564f-d0a4-47b0-8fc1-a6f83e714112
let
	fig = Figure(size = (1100, 360))
	ax = Axis(fig[1,1]; title = "Diffusion-map eigenvalues |λₖ|, α = $α_choice",
	  xlabel = "k", ylabel = "|λₖ|")
	λabs = abs.(Float64.(dm.λ))
	n = min(15, length(λabs))
	scatterlines!(ax, 1:n, λabs[1:n]; color = :teal, markersize = 8)
	gaps = abs.(diff(λabs[1:n]))
	gap_after = argmax(gaps)
	vlines!(ax, [gap_after + 0.5]; color = :gray, linestyle = :dash,
	  label = "largest gap after k=$gap_after (Δ = $(round(maximum(gaps); digits=3)))")
	axislegend(ax; position = :rt, framevisible = false)
	fig
end

# ╔═╡ 36b45df3-bb5c-4739-a21c-7f06d07ff491
md"""
Eigenvalues measure how strongly graph modes persist under repeated applications of P. The plotted spectrum includes the constant mode at λ₁ = 1; the automatic largest-gap marker can therefore select the gap between that mode and the first informative coordinate. Gaps among nonconstant modes can guide a representation's truncation, with sensitivity to bandwidth, normalization, and sampling. Assigning a physical dimension or naming individual climate modes requires evidence beyond this spectrum.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000009
md"""
### What the kernel actually sees: the affinity matrix

The diffusion-map construction starts with the pairwise affinity
matrix
```math
K_\varepsilon(i, j) = \exp(-\|x_i - x_j\|^2 / \varepsilon),
```
a $T \times T$ object telling you "how similar is month $i$ to month
$j$." Below is that matrix as an image, with months ordered
chronologically on both axes. Bright off-diagonal patches mark months
in different decades that nevertheless share a similar SST anomaly
configuration. The leading eigenvectors of (the α-renormalised,
row-stochastic version of) this matrix become Ψ₂, Ψ₃, ...
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000a
let
	D2 = pairwise_sqdist(X)
	ε_use = median_bandwidth(D2)
	K_aff = exp.(.-(D2 ./ ε_use))
	years_axis = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1)/12 for t in time_w]

	fig = Figure(size = (820, 740))
	ax = Axis(fig[1, 1];
			  title = "Pairwise affinity K_ε(month_i, month_j), tropical-Pacific SST",
			  xlabel = "month i (year)", ylabel = "month j (year)",
			  aspect = 1.0)
	hm = heatmap!(ax, years_axis, years_axis, K_aff;
				  colormap = :viridis, colorrange = (0, 1))
	Colorbar(fig[1, 2], hm; label = "kernel similarity")
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000b
md"""
### Kernel neighbors of December 1997

The Gaussian kernel ranks other months by similarity of their SST features to December 1997, excluding the query itself. Displaying the ten closest matches in diffusion coordinates connects input-space affinities with the learned representation.
"""

# ╔═╡ c5eb626e-d6ff-473a-8adc-92ba2cdb10b1
md"""
### Diffusion coordinates at α = 1 and a selected trajectory

This view is fixed at α = 1 regardless of the slider. Each point represents a month. The trajectory connects selected months in temporal order after fitting the representation.
"""

# ╔═╡ 1e58cbed-8302-4c65-a79c-5ea3fabc1c1c
let
	# Use α = 1 specifically (the most physically meaningful) regardless of widget
	dm1 = fit_diffusion_map(X; α=1.0, d=10, t=1)
	Ψ1 = Float64.(dm1.coords)
	for k in 1:5
		if cor(Ψ1[valid_n, k], nino34[valid_n]) < 0
			Ψ1[:, k] .= -Ψ1[:, k]
		end
	end

	fig = Figure(size = (1300, 480))

	ax1 = Axis(fig[1,1]; title = "Phase portrait Ψ₂ vs Ψ₃, all 900 months (α=1)",
	  xlabel = "Ψ₂", ylabel = "Ψ₃")
	sct = scatter!(ax1, Ψ1[:, 1], Ψ1[:, 2]; color = nino34, colormap = :RdBu_11,
	  colorrange = (-2.5, 2.5), markersize = 5, alpha = 0.7)
	Colorbar(fig[1, 2], sct; label = "Niño 3.4 (°C)")

	for d in [Date(1972,12,1), Date(1982,12,1), Date(1997,12,1),
	     Date(2015,12,1), Date(2023,12,1)]
		idx = findfirst(==(d), Date.(time_w))
		if !isnothing(idx)
			scatter!(ax1, [Ψ1[idx, 1]], [Ψ1[idx, 2]];
			  color = :black, markersize = 14, marker = :star5)
			text!(ax1, Ψ1[idx, 1] + 0.005, Ψ1[idx, 2];
			  text = " " * Dates.format(d, "u yyyy"),
			  fontsize = 9, color = :black)
		end
	end

	# 1996-1999 trajectory
	ax2 = Axis(fig[1,3];
	  title = "Trajectory through the 1996–1999 ENSO cycle",
	  xlabel = "Ψ₂", ylabel = "Ψ₃")
	cs = findfirst(==(Date(1996,1,1)), Date.(time_w))
	ce = findfirst(==(Date(1999,12,1)), Date.(time_w))
	seg = cs:ce
	scatter!(ax2, Ψ1[seg, 1], Ψ1[seg, 2]; color = nino34[seg],
	  colormap = :RdBu_11, colorrange = (-2.5, 2.5), markersize = 8)
	lines!(ax2, Ψ1[seg, 1], Ψ1[seg, 2]; color = (:black, 0.4), linewidth = 1.0)
	fig
end

# ╔═╡ ba48d31e-046a-48d5-a9e4-ac4117dd4855
md"""
The left panel places months in two diffusion coordinates and colors them by Niño 3.4. The right panel follows the selected 1996–1999 interval through those coordinates.

These views locate warm and cool SST states and trace their observed sequence. Their shape depends on the coordinate functions: nonlinear eigenfunctions can bend an embedding, and linear PCA projections can also display curved data. Interpreting Ψ₃ as a physical asymmetry coordinate or the trajectory as a specific oscillator therefore requires a separate test.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000d
md"""
### Occupancy in the fitted coordinates

A two-dimensional histogram, smoothed with a local box filter, shows where observed months concentrate in the fitted coordinates. This empirical occupancy view connects graph geometry with uneven sampling across states.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000f
md"""
### Calendar timing along a selected trajectory

Coloring the 1996–1999 trajectory by month of year connects position in the embedding with seasonal timing. This selected interval provides context for a broader seasonal phase-locking analysis.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000001
md"""
### Three-coordinate and spatial views

Adding a third coordinate exposes variation hidden by the two-dimensional projection. The stored figures and animations connect positions in the embedding to SST maps and the reference series.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000002
PlutoUI.LocalResource(joinpath(@__DIR__, "..", "figures", "fig11_attractor3d.png"))

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000003
md"""
*Figure 11.* Monthly fields in three diffusion coordinates, with selected historical events marked. This is a higher-dimensional view of the fitted representation.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000004
PlutoUI.LocalResource(joinpath(@__DIR__, "..", "figures", "fig12_trajectory_poster.png"))

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000005
md"""
*Figure 12.* The January 1998 state in the embedding, paired with the reference time series. The stored animation follows the historical sequence through these coordinates.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000006
PlutoUI.LocalResource(joinpath(@__DIR__, "..", "figures", "fig13_sst_poster.png"))

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-500000000007
md"""
*Existing figure 13.* The SST anomaly field for the selected January 1998 month. The dashed box marks the Niño 3.4 region. The map is the spatial view of the state highlighted above.
"""

# ╔═╡ a33add4b-3acc-48c1-bc0d-97c814e3b39f
md"""
## 5, Method comparison

The side-by-side view compares PCA with diffusion maps at α = 0, 0.5, and 1, using the same monthly features and Niño 3.4 coloring. It separates changes in coordinate geometry from changes in leading-coordinate index agreement.
"""

# ╔═╡ dddbef5c-830b-4c67-816d-3e0d21e2288b
begin
	dm_alpha = Dict()
	for α in (0.0, 0.5, 1.0)
		d = fit_diffusion_map(X; α=α, d=10, t=1)
		Ψα = Float64.(d.coords)
		for k in 1:5
			if cor(Ψα[valid_n, k], nino34[valid_n]) < 0
				Ψα[:, k] .= -Ψα[:, k]
			end
		end
		dm_alpha[α] = (dm=d, Ψ=Ψα,
		        ρ = [cor(Ψα[valid_n, k], nino34[valid_n]) for k in 1:5])
	end
	"computed all three α"
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000c
let
	dec97_idx = findfirst(==(Date(1997, 12, 1)), Date.(time_w))
	α1 = dm_alpha[1.0]
	Ψ1 = α1.Ψ

	D2 = pairwise_sqdist(X)
	ε_use = median_bandwidth(D2)
	sim = exp.(-D2[dec97_idx, :] ./ ε_use)
	sim[dec97_idx] = -1.0
	top10 = sortperm(sim; rev = true)[1:10]

	fig = Figure(size = (1100, 620))
	ax = Axis(fig[1, 1];
			  title = "Top-10 months most similar to Dec 1997 (Gaussian kernel, α=1 embedding)",
			  xlabel = "Ψ₂", ylabel = "Ψ₃")

	scatter!(ax, Ψ1[:, 1], Ψ1[:, 2]; color = (:gray, 0.25), markersize = 4)
	scatter!(ax, Ψ1[top10, 1], Ψ1[top10, 2];
			 color = :crimson, markersize = 14,
			 strokecolor = :black, strokewidth = 0.5)
	scatter!(ax, [Ψ1[dec97_idx, 1]], [Ψ1[dec97_idx, 2]];
			 color = :black, markersize = 24, marker = :star5)
	text!(ax, Ψ1[dec97_idx, 1] - 0.02, Ψ1[dec97_idx, 2] - 0.04;
		  text = "Dec 1997 (query)", fontsize = 12, font = :bold,
		  align = (:right, :top))

	# Rather than labelling every point (which overlap badly when neighbours
	# cluster), list the 10 months in an annotation box. The visual point is
	# the *clustering*; the identity of each month is secondary.
	months_str = join([Dates.format(Date(time_w[i]), "u yyyy") for i in top10], ", ")
	# Wrap to two lines around the comma after the 5th entry.
	parts = split(months_str, ", ")
	line1 = join(parts[1:5], ", ")
	line2 = join(parts[6:end], ", ")
	xmin, xmax = extrema(Ψ1[:, 1])
	ymax = maximum(Ψ1[:, 2])
	text!(ax, xmin + 0.02, ymax;
		  text = "10 nearest neighbours of Dec 1997:\n" * line1 * ",\n" * line2,
		  fontsize = 11, font = :regular, align = (:left, :top))
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa0000000e
let
	α1 = dm_alpha[1.0]
	Ψ1 = α1.Ψ
	xs = Ψ1[:, 1]; ys = Ψ1[:, 2]

	nb = 36
	xrng = range(minimum(xs) - 0.02, maximum(xs) + 0.02; length = nb + 1)
	yrng = range(minimum(ys) - 0.02, maximum(ys) + 0.02; length = nb + 1)
	Hd = zeros(nb, nb)
	for k in 1:length(xs)
		i = clamp(searchsortedlast(xrng, xs[k]), 1, nb)
		j = clamp(searchsortedlast(yrng, ys[k]), 1, nb)
		Hd[i, j] += 1
	end
	# Smooth the histogram with a 3x3 box filter for cleaner contours.
	Hs = copy(Hd)
	for j in 2:nb-1, i in 2:nb-1
		Hs[i, j] = (Hd[i-1, j-1] + Hd[i, j-1] + Hd[i+1, j-1] +
		            Hd[i-1, j  ] + Hd[i, j  ] + Hd[i+1, j  ] +
		            Hd[i-1, j+1] + Hd[i, j+1] + Hd[i+1, j+1]) / 9
	end
	xc = (xrng[1:end-1] .+ xrng[2:end]) ./ 2
	yc = (yrng[1:end-1] .+ yrng[2:end]) ./ 2

	fig = Figure(size = (950, 700))
	ax = Axis(fig[1, 1];
			  title = "Phase-portrait state density + Niño 3.4-coloured points",
			  xlabel = "Ψ₂", ylabel = "Ψ₃",
			  backgroundcolor = :white)
	# Levels start above 0 so empty bins stay blank instead of painting the
	# whole axis dark; ramp through warm tones so density reads at a glance.
	hmax = maximum(Hs)
	contourf!(ax, xc, yc, Hs;
			  colormap = cgrad(:matter, rev = false),
			  levels = range(0.4, hmax; length = 7),
			  extendlow = :transparent)
	sct = scatter!(ax, xs, ys; color = nino34, colormap = :RdBu_11,
				   colorrange = (-2.5, 2.5), markersize = 5,
				   strokecolor = (:black, 0.5), strokewidth = 0.3)
	Colorbar(fig[1, 2], sct; label = "Niño 3.4 (°C)")
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000010
let
	α1 = dm_alpha[1.0]
	Ψ1 = α1.Ψ
	cs = findfirst(==(Date(1996, 1, 1)), Date.(time_w))
	ce = findfirst(==(Date(1999, 12, 1)), Date.(time_w))
	seg = cs:ce
	months_seg = [Dates.month(Date(t)) for t in time_w[seg]]

	fig = Figure(size = (1000, 700))
	ax = Axis(fig[1, 1];
			  title = "1996–1999 trajectory, coloured by month-of-year (seasonal phase)",
			  xlabel = "Ψ₂", ylabel = "Ψ₃")
	scatter!(ax, Ψ1[:, 1], Ψ1[:, 2]; color = (:gray, 0.18), markersize = 3)
	lines!(ax, Ψ1[seg, 1], Ψ1[seg, 2]; color = (:black, 0.4), linewidth = 1.0)
	sct = scatter!(ax, Ψ1[seg, 1], Ψ1[seg, 2]; color = months_seg,
				   colormap = :twilight, colorrange = (1, 12), markersize = 11,
				   strokecolor = :black, strokewidth = 0.3)
	Colorbar(fig[1, 2], sct; label = "month of year",
			 ticks = ([1, 4, 7, 10, 12], ["Jan", "Apr", "Jul", "Oct", "Dec"]))

	for d in [Date(1996, 12, 1), Date(1997, 12, 1),
			  Date(1998, 12, 1), Date(1999, 12, 1)]
		idx = findfirst(==(d), Date.(time_w))
		if !isnothing(idx)
			text!(ax, Ψ1[idx, 1] + 0.008, Ψ1[idx, 2];
				  text = "Dec " * string(Dates.year(d)),
				  fontsize = 9, color = :black)
		end
	end
	fig
end

# ╔═╡ 843da5fd-0fb0-499a-8807-e2a09bb2d20c
let
	fig = Figure(size = (1500, 380))
	ax1 = Axis(fig[1, 1]; title = "PCA: PC1 vs PC2", xlabel = "PC 1", ylabel = "PC 2")
	sct = scatter!(ax1, PC[:, 1], PC[:, 2]; color = nino34, colormap = :RdBu_11,
	  colorrange = (-2.5, 2.5), markersize = 4, alpha = 0.7)
	for (j, α) in enumerate((0.0, 0.5, 1.0))
		ax2 = Axis(fig[1, j+1]; title = "DMAP α = $α: Ψ₂ vs Ψ₃",
		  xlabel = "Ψ₂", ylabel = "Ψ₃")
		Ψα = dm_alpha[α].Ψ
		scatter!(ax2, Ψα[:, 1], Ψα[:, 2]; color = nino34, colormap = :RdBu_11,
		  colorrange = (-2.5, 2.5), markersize = 4, alpha = 0.7)
	end
	Colorbar(fig[1, 5], sct; label = "Niño 3.4 (°C)")
	fig
end

# ╔═╡ 67db4c45-9a70-4304-b9d9-c05ec77537ff
md"""
The saved correlations in `figures/enso_metrics.txt` are 0.946 for PCA and 0.943, 0.937, and 0.923 for diffusion maps at α = 0, 0.5, and 1. The leading-coordinate metric favors PCA across these settings. The α = 1 rationale targets density correction rather than maximizing this correlation.

The table below is computed with the default median bandwidth and diffusion time 1; the α and time controls above affect the separately selected embedding.

| method | same-month correlation with Niño 3.4 |
|:---|:---:|
| PCA (PC1) | $(round(ρ_PC[1]; digits=3)) |
| DMAP α=0 | $(round(dm_alpha[0.0].ρ[1]; digits=3)) |
| DMAP α=0.5 | $(round(dm_alpha[0.5].ρ[1]; digits=3)) |
| DMAP α=1 | $(round(dm_alpha[1.0].ρ[1]; digits=3)) |

**Higher coordinates.** The saved α = 1 absolute correlations for Ψ₂–Ψ₆ are 0.923, 0.297, 0.196, 0.142, and 0.072. This checks how index agreement is distributed across graph modes. The same leading-five comparison is computed for PCA and each α setting; `figures/fig05_correlations.png` contains the saved mode-by-mode view.

### Bandwidth sensitivity

`src/enso_figures.jl` also fits α = 1 maps at 11 logarithmically spaced bandwidth multipliers ``10^s``, with ``s`` from −1 to 1.5, relative to the median-distance bandwidth. It compares absolute leading-coordinate correlation with the PCA baseline and saves `figures/fig08_bandwidth.png`. This probes dependence on the kernel's locality scale. The original report describes reduced correlation at small bandwidths and a broad high-bandwidth plateau; the sweep's numerical values are stored graphically rather than in the saved metric table, so the exact plateau and connectivity claims remain unverified. These full-sample comparisons characterize sensitivity rather than held-out parameter selection.

### Additional representation utilities

The source also provides PCA low-rank reconstruction (`denoise_pca`), graph smoothing by repeated multiplication with the fitted transition matrix (`diffusion_smooth`), and embedding-quality utilities in `src/metrics.jl`. Trustworthiness penalizes neighbors introduced by an embedding; continuity penalizes original neighbors lost in it. Orthogonal Procrustes alignment fits a rotation/reflection between coordinate matrices and reports relative residual error. These are implemented tools for reconstruction, smoothing, and geometric comparison; the notebook contains no evaluated results from these utilities.

### Nyström extension: implemented projection machinery

`nystrom_extend` in `src/diffusion.jl` maps new feature rows into a fitted diffusion representation. For a new row ``y``, it computes affinities to the stored training anchors using the fitted bandwidth, estimates ``q(y)=\sum_iK(y,x_i)``, applies the same α normalization using the stored training ``q_i``, and row-normalizes to ``\widetilde P(y,x_i)``. It then extends each eigenfunction through
```math
\psi_j(y)=\frac{1}{\lambda_j}\sum_i\widetilde P(y,x_i)\psi_j(x_i),
\qquad \Phi_t(y)_j=\lambda_j^t\psi_j(y),\quad j\ge2.
```
This is the implemented route to coordinates for new observations without refitting the eigensystem. It uses all fitted anchors, rather than selecting a landmark subset. The notebook's results use full-sample fitting; the extension remains an implementation capability without an evaluated temporal-holdout result. Such evaluation would also need training-only preprocessing and checks of numerical behavior for very small eigenvalues and weak affinities.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-400000000001
md"""
## 5.5, Linear Inverse Model: from representations to temporal dynamics

PCA and snapshot diffusion maps organize fields by variance or similarity. LIM adds the order of observations by fitting lagged covariance in the top 10 PC coordinates at a one-month lag. The motivating model is a stationary stochastic linear system,
```math
\frac{dz}{dt}=Lz+Q\xi(t).
```
With a consistent column-state convention, ``C(\tau)=E[z_{t+\tau}z_t^\top]`` gives ``G(\tau)=C(\tau)C(0)^{-1}`` and ``L=\log G/\tau``. A complex eigenvalue ``\sigma=a+i\omega`` describes a mode with period ``2\pi/|\omega|`` and, for ``a<0``, decay time ``-1/a``.

The present `src/lim.jl` computes ``C_0=X_c^\top X_c/(T-1)``, ``C_1=X_a^\top X_b/(T-\tau-1)``, and ``G=C_1C_0^{-1}``, where rows of ``X_a`` precede rows of ``X_b``. This lag-covariance orientation differs from the column-state convention above. The propagator orientation and eigenmode projection need a consistent convention check before physical interpretation of the fitted mode coordinates.

The notebook selects the complex mode with positive frequency and smallest absolute real part among periods of 2–8 years. This period window is part of the selection rule. LIM therefore supplies a model-dependent temporal summary that complements the snapshot embeddings; a prospective forecast comparison would require held-out trajectories.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-400000000003
let
	fig = Figure(size = (1300, 360))
	# Eigenvalues in complex plane
	ax1 = Axis(fig[1, 1];
	  title = "LIM eigenvalues σ, period = $(round(lim_periods[lim_idx]; digits=2)) yr, e-fold = $(round(lim_decays[lim_idx]; digits=2)) yr",
	  xlabel = "Re σ (1/month), damping",
	  ylabel = "Im σ (1/month), frequency",
	  aspect = 1.0)
	scatter!(ax1, real.(lim.σ), imag.(lim.σ); color = :gray, markersize = 12)
	scatter!(ax1,
	  [real(lim.σ[lim_idx]), real(lim.σ[lim_idx])],
	  [imag(lim.σ[lim_idx]), -imag(lim.σ[lim_idx])];
	  color = :crimson, markersize = 16, marker = :star5,
	  label = "ENSO mode")
	vlines!(ax1, [0]; color = :gray, linewidth = 0.7)
	hlines!(ax1, [0]; color = :gray, linewidth = 0.7)
	axislegend(ax1; position = :rb, framevisible = false)

	# Time series
	years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]
	ax2 = Axis(fig[1, 2];
	  title = "LIM ENSO mode Re(z) vs Niño 3.4  (ρ = $(round(ρ_LIM; digits=3)))",
	  xlabel = "year", ylabel = "anomaly")
	s = std(nino34[valid_n]) / std(z_re)
	lines!(ax2, years, nino34;   color = (:steelblue, 0.85), label = "Niño 3.4")
	lines!(ax2, years, z_re .* s; color = (:darkorange, 0.85), label = "Re(z) LIM ENSO mode")
	hlines!(ax2, [0]; color = :gray, linewidth = 0.5)
	axislegend(ax2; position = :rt, framevisible = false)
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-400000000004
md"""
The active selected mode has period $(round(lim_periods[lim_idx]; digits=2)) years and decay time $(round(lim_decays[lim_idx]; digits=2)) years.

`figures/lim_summary.txt` records 3.12 years, 0.68 years, and correlation 0.613 between the real mode projection and Niño 3.4. Period and decay summarize the fitted lag-covariance operator. The real part of a complex coordinate depends on the mode's phase convention, so its index correlation also depends on how that projection is defined. These distinctions, together with the covariance-orientation issue, matter when comparing LIM with PCA.
"""

# ╔═╡ 88921a71-c8a1-46dd-8d51-fb38d78d6b24
md"""
## 6, Controls: temporal alignment and spatial structure

The independent-month permutation control holds the fitted embeddings fixed, shuffles the valid Niño 3.4 values 500 times with a fixed random seed, and recomputes absolute correlation. It preserves the reference's marginal distribution while removing its month-by-month alignment with the learned coordinate.

Shuffling also removes the reference's serial dependence. For persistent climate time series this changes the comparison distribution and can understate the variability of chance alignment relative to a dependence-preserving null. The experiment is useful as an alignment diagnostic; time-series significance would require a separately specified block or surrogate procedure that respects temporal dependence.
"""

# ╔═╡ 7e12b7a2-b22a-4cb2-80cd-566b17454b2d
begin
	rng = MersenneTwister(2026)
	n_clean = nino34[valid_n]
	chance_PC = Float64[]
	chance_DM = Float64[]
	for _ in 1:500
		p = shuffle(rng, n_clean)
		push!(chance_PC, abs(cor(PC[valid_n, 1], p)))
		push!(chance_DM, abs(cor(Ψ[valid_n, 1], p)))
	end
	"null distribution computed (n=500)"
end

# ╔═╡ 1a1cfdf1-070e-47ab-9647-2f97f2fa1625
let
	fig = Figure(size = (1100, 360))
	ax = Axis(fig[1, 1];
	  title = "Independent-month shuffle, n=500: temporal dependence not preserved",
	  xlabel = "|ρ|", ylabel = "count")
	hist!(ax, chance_PC; bins = 30, color = (:crimson, 0.5), label = "PCA shuffle")
	hist!(ax, chance_DM; bins = 30, color = (:teal, 0.5),  label = "DMAP shuffle")
	vlines!(ax, [ρ_PC[1]]; color = :crimson, linewidth = 3,
	  label = "real PC 1 = $(round(ρ_PC[1]; digits=3))")
	vlines!(ax, [ρ_Ψ[1]];  color = :teal,  linewidth = 3,
	  label = "real Ψ₂ = $(round(ρ_Ψ[1]; digits=3))")
	axislegend(ax; position = :rt, framevisible = false)
	σ_PC = (ρ_PC[1] - mean(chance_PC)) / std(chance_PC)
	σ_DM = (ρ_Ψ[1] - mean(chance_DM)) / std(chance_DM)
	text!(ax, 0.5, 0.95; text = "PCA: shuffle ignores temporal dependence",
	  space = :relative, fontsize = 12, color = :crimson)
	text!(ax, 0.5, 0.88; text = "DMAP: not a time-series significance test",
	  space = :relative, fontsize = 12, color = :teal)
	fig
end

# ╔═╡ c38ddd37-0dfc-43fa-bf05-b6e5cea6b9e2
md"""
The saved shuffle means are 0.0261 (PCA) and 0.0265 (diffusion maps), with 95th percentiles 0.0615 and 0.0659. The aligned correlations exceed these shuffled values. The archived 46σ and 43σ separations describe this particular independent-month distribution; temporal persistence prevents treating them as calibrated time-series significance levels.

**Spatial-shuffle control.** `src/enso_analysis.jl` independently permutes feature order within each monthly row, then refits PCA and diffusion maps. This preserves each row's distribution of weighted feature values while disrupting consistent geographic correspondence between months. It asks whether representation–index agreement survives when spatial organization is removed. The original report describes residual correlation around 0.65, but the current saved metric file omits separate spatial-shuffle estimates, so that approximate report-level value has incomplete numerical provenance. A difference between correlations cannot partition the physical signal into spatial and amplitude contributions.
"""

# ╔═╡ 7a9ae0e7-0382-4f81-aab0-b8b42bbab1e8
md"""
## 7, Time-delay features: testing temporal context

A snapshot gives SST at one time; a delay vector also describes the recent path to that state:
```math
y_t=[x_t,x_{t-\tau},\ldots,x_{t-k\tau}].
```
`delay_stack` returns ``T-k\tau`` rows and ``(k+1)N`` columns, together with the original index of each row's current month. The experiment uses monthly spacing ``\tau=1`` and ``k\in\{0,3,6,9,12,18,24\}``, then fits diffusion maps at α = 1 and diffusion time 1. Its median-distance bandwidth is recomputed in each feature space. The index is aligned to the current month at the end of each history window.

Takens-style reconstruction motivates the inclusion of history: under smooth deterministic dynamics and generic-observation assumptions, sufficiently many delayed observations can encode an underlying state. Here multivariate, noisy SST histories are an empirical feature construction, and the relevant question is whether they improve the chosen representation metric.

Two feature pipelines are evaluated in `src/enso_figures.jl`: delays of the full weighted SST field, and delays of the top 10 PCA scores. The latter reduces each monthly block before adding history. The notebook computation below performs the raw-field sweep; saved metrics include both pipelines. Each compares the leading coordinate with the maximum absolute correlation among the first **five** nonconstant coordinates (Ψ₂–Ψ₆), to check whether index agreement is concentrated elsewhere in the spectrum.
"""

# ╔═╡ 6a917333-1270-4ead-93c9-7bba4fc1b6fe
begin
	delay_ks = [0, 3, 6, 9, 12, 18, 24]
	ρ_delay = Float64[]
	ρ_delay_max = Float64[]
	for kd in delay_ks
		if kd == 0
			Y = X
			src = collect(1:size(Y,1))
		else
			Y, src = delay_stack(X, kd)
		end
		d = fit_diffusion_map(Y; α=1.0, d=6, t=1)
		Ψd = Float64.(d.coords)
		aligned = nino34[src]
		valid = .!isnan.(aligned)
		cs = [abs(cor(Ψd[valid, k], aligned[valid])) for k in 1:5]
		push!(ρ_delay, cs[1])
		push!(ρ_delay_max, maximum(cs))
	end
	"delay sweep done"
end

# ╔═╡ 1706f3c3-254d-408e-b635-739f9313185a
let
	fig = Figure(size = (1100, 380))
	ax = Axis(fig[1,1];
	  title = "Time-delay embedding: leading-mode correlation",
	  xlabel = "delay k (months)", ylabel = "|ρ(Ψ₂', Niño 3.4)|")
	scatterlines!(ax, delay_ks, ρ_delay; color = :crimson, markersize = 10,
	  label = "Ψ₂' (leading nontrivial)")
	scatterlines!(ax, delay_ks, ρ_delay_max; color = :teal, markersize = 10,
	  label = "max over Ψ₂..Ψ₆")
	hlines!(ax, [ρ_PC[1]]; color = :gray, linestyle = :dash,
	  label = "PCA baseline")
	axislegend(ax; position = :rb, framevisible = false)
	fig
end

# ╔═╡ 00a1d8c0-29e1-48f2-bb0c-c4902f9fd77c
begin
	delay_diff_str = string(round(ρ_delay[findfirst(==(12), delay_ks)] - ρ_delay[1]; digits=3))
	delay_first_str = string(round(ρ_delay[1];  digits=3))
	delay_last_str = string(round(ρ_delay[end]; digits=3))
	md"""
The active raw-field sweep gives a change of $(delay_diff_str) at 12 lags relative to no delay; its first and last leading-coordinate correlations are $(delay_first_str) and $(delay_last_str).

**Saved comparison** (`figures/enso_metrics.txt`; absolute correlations):

| lags k | raw leading | raw best of five | top-10-PC leading | top-10-PC best of five |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.923 | 0.923 | 0.909 | 0.909 |
| 3 | 0.868 | 0.868 | 0.854 | 0.854 |
| 6 | 0.755 | 0.755 | 0.742 | 0.742 |
| 9 | 0.613 | 0.613 | 0.601 | 0.601 |
| 12 | 0.470 | 0.693 | 0.459 | 0.631 |
| 18 | 0.253 | 0.692 | 0.242 | 0.700 |
| 24 | 0.136 | 0.605 | 0.129 | 0.601 |

The predicted improvement of at least 0.05 at 12 lags was unsuccessful in both pipelines: the saved changes are −0.453 (raw) and −0.450 (PC space). Leading-coordinate agreement decreases throughout the tested sweep. At longer delays, a different leading-five coordinate has higher correlation, showing that a single leading-coordinate score misses some retained index agreement. Even that selected maximum remains below the corresponding no-delay result.

Several effects are entangled: delay stacking changes the distance metric, bandwidth, number of observations, and dates being compared. The best-of-five value also uses the reference to select a coordinate, making it an exploratory upper comparison rather than an independently evaluated score. Distance concentration and eigenfunction reordering are possible mechanisms; the saved correlations alone do not identify their causes. The original report additionally describes sparse spacings τ = 3 and 6 as unsuccessful, but the inspected sweep code and metric file retain only τ = 1, leaving the sparse-spacing claim without reproducible settings or saved values. The supported negative result concerns these specific feature constructions and same-month metric.
"""
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-200000000001
md"""
## 7.5, Temporal context: wavelet view of PC1

A Morlet continuous wavelet transform describes how PC1's frequency content varies over time. `src/spectrogram.jl` computes the transform using FFT convolution. The plot divides wavelet power by PC1 variance and marks a 3–7 year reference band. This connects the learned spatial pattern to the time scales present in its score series, with boundary interpretation limited by the cone of influence.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-200000000003
let
	years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]
	σ² = var(PC[:, 1])
	P_norm = P_pc ./ σ²

	fig = Figure(size = (1300, 380))
	ax = Axis(fig[1, 1];
	  title = "Morlet wavelet power spectrum of PC 1 (variance-normalised)",
	  xlabel = "year", ylabel = "period (years)",
	  yscale = log2,
	  yticks = ([0.5, 1, 2, 3, 5, 7, 10, 16],
	       ["0.5", "1", "2", "3", "5", "7", "10", "16"]))
	hm = heatmap!(ax, years, periods_pc, P_norm;
	  colormap = :viridis, colorrange = (0, quantile(vec(P_norm), 0.98)))
	Colorbar(fig[1, 2], hm; label = "|W(t,s)|² / σ²")
	hlines!(ax, [3, 7]; color = :white, linestyle = :dash, linewidth = 1.0)
	text!(ax, years[1] + 1, 4.5; text = "ENSO band (3–7 yr)",
	   fontsize = 11, color = :white)
	# cone of influence, outside region is unreliable
	band!(ax, years, coi_pc, fill(maximum(periods_pc), length(years));
	   color = (:black, 0.25))
	fig
end

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-200000000004
md"""
The wavelet view locates periods and epochs contributing power to PC1. The shaded cone-of-influence region identifies scales affected by the finite record's boundaries. It provides a descriptive time-frequency complement to LIM's fitted period and decay.
"""

# ╔═╡ 5239ef9c-6555-433a-a749-79cf136aff0e
md"""
## 8, Summary

- **Representation learning:** calendar-month anomalies and spatial features feed PCA and diffusion maps. PCA's saved leading variance fraction is 0.401, with Niño 3.4 correlation 0.946; diffusion maps at α = 1 gives 0.923. PCA is the stronger baseline on this same-month metric.
- **Sensitivity:** the α-family gives saved correlations 0.943, 0.937, and 0.923. Bandwidth and diffusion-time comparisons examine how graph construction and spectral weighting shape the representation.
- **Temporal features, including the negative result:** raw-field and top-10-PC delay pipelines both lose leading-coordinate agreement as history length increases. At 12 lags the saved correlations are 0.470 and 0.459. Looking across the first five nonconstant modes recovers some agreement, while remaining below each no-delay baseline.
- **Dynamics and diagnostics:** LIM fits a lagged-covariance model; wavelets describe PC1's time-frequency content. Label permutations and spatial shuffling probe alignment and spatial organization, with different methodological limits.
- **Extension machinery:** PCA projection and Nyström diffusion-coordinate extension are implemented. Their role in a temporal holdout remains an evaluation task.

The evidence supports useful low-dimensional summaries of ENSO-related SST variation in the fitted record. Geometry plots motivate further questions about higher coordinates; demonstrated nonlinear superiority and physical attractor identification remain open. Priority technical checks are correcting area-weight ordering and verifying LIM conventions, followed by re-evaluation. A temporal holdout would fit climatology, masks, centering, representations, and parameter choices on training data, then project later months; a dependence-aware control would address serial correlation. Both are future evaluation designs.
"""

# ╔═╡ 30d4a0b5-bbbb-cccc-dddd-aaaa00000020
md"""
## 9, Works cited

Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007).
El Niño Modoki and its possible teleconnection.
*Journal of Geophysical Research*, **112**, C11007.

Bjerknes, J. (1969). Atmospheric teleconnections from the equatorial
Pacific. *Monthly Weather Review*, **97**(3), 163–172.

Burgers, G., Jin, F.-F., & van Oldenborgh, G. J. (2005).
The simplest ENSO recharge oscillator.
*Geophysical Research Letters*, **32**, L13706.

Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
*Applied and Computational Harmonic Analysis*, **21**(1), 5–30.

Eckart, C., & Young, G. (1936). The approximation of one matrix by
another of lower rank. *Psychometrika*, **1**(3), 211–218.

Huang, B., Thorne, P. W., Banzon, V. F., Boyer, T., Chepurin, G.,
Lawrimore, J. H., et al. (2017). Extended Reconstructed Sea Surface
Temperature, Version 5 (ERSSTv5): Upgrades, validations, and
intercomparisons.
*Journal of Climate*, **30**(20), 8179–8205.

Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO.
Part I: Conceptual model.
*Journal of the Atmospheric Sciences*, **54**(7), 811–829.

Lorenz, E. N. (1956). Empirical orthogonal functions and statistical
weather prediction.
*Statistical Forecasting Project Scientific Report No. 1*,
MIT Department of Meteorology.

Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013).
Context-dependent computation by recurrent dynamics in prefrontal
cortex. *Nature*, **503**(7474), 78–84.

Pandarinath, C., O'Shea, D. J., Collins, J., Jozefowicz, R.,
Stavisky, S. D., Kao, J. C., et al. (2018). Inferring single-trial
neural population dynamics using sequential auto-encoders.
*Nature Methods*, **15**(10), 805–815.

Penland, C., & Sardeshmukh, P. D. (1995). The optimal growth of
tropical sea surface temperature anomalies.
*Journal of Climate*, **8**(8), 1999–2024.

Takens, F. (1981). Detecting strange attractors in turbulence. In
*Dynamical Systems and Turbulence, Warwick 1980*
(Lecture Notes in Mathematics, vol. 898, pp. 366–381). Springer.

Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet
analysis. *Bulletin of the American Meteorological Society*,
**79**(1), 61–78.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CairoMakie = "~0.15.10"
ColorSchemes = "~3.31.0"
Distances = "~0.10.12"
FFTW = "~1.10.0"
NCDatasets = "~0.14.15"
NearestNeighbors = "~0.4.27"
PlutoUI = "~0.7.80"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "f59122545482c2257bac1f27666956e0a1325a64"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CFTime]]
deps = ["Dates", "Printf"]
git-tree-sha1 = "b2c9f2d3b8014323c0a8f2b401b68fa6bb08ed06"
uuid = "179af706-886a-5703-950a-314cd64e0468"
version = "0.2.8"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "bf2d9cd1ec0c4ce3e0b5aaad192074969413f626"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.10"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d0efe2c6fdcdaa1c161d206aa8b933788397ec71"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.6+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonDataModel]]
deps = ["CFTime", "DataStructures", "Dates", "DiskArrays", "Preferences", "Printf", "Statistics"]
git-tree-sha1 = "bf07704e843daabd2cb2bb1404571656f80bce16"
uuid = "1fbeeb36-5f17-413c-809b-666fb144f157"
version = "0.4.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "3b4be73db165146d8a88e47924f464e55ab053cd"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.7"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.DiskArrays]]
deps = ["ConstructionBase", "LRUCache", "Mmap", "OffsetArrays"]
git-tree-sha1 = "e5d9ce1b751ddf9bcd9d36b51249dce8ea73cd55"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.4.19"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e421c1938fafab0165b04dc1a9dbe2a26272952c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.125"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8f05e9a2e7c2e3eb524102bb2926c5743c07fbe1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.0+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "aws_c_s3_jll", "dlfcn_win32_jll", "libaec_jll", "mpif_jll"]
git-tree-sha1 = "45337643a2d97262d5fe72ce1f13e8a662d13d62"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.1.2+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "baaaebd42ed9ee1bd9173cfd56910e55a8622ee1"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.13.0+1"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "3e6273749a2df3a5c9067657510ad01ba5039a92"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.8"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "fe23330af47b8ab4e135b2ff65f7398c3a2bfc65"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.2"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "4260cfc991b8885bf747801fb60dd4503250e478"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.11"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPIABI_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9be143b6045719e8fb019d2b3bc2aebad1184fef"
uuid = "b5ada748-db0f-5fc0-8972-9331c762740c"
version = "0.1.5+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "07dbec8aab01696edc0151a401a6cdfe95b9b885"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "5.0.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "675df097f8eeb28998b2cfe3b25655af73d5f7df"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.6+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "0708c6a1f3cb18ba6482c4174058084c8d6deaf4"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.10"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "7eb8cdaa6f0e8081616367c10b31b9d9b34bb02a"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.7"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NCDatasets]]
deps = ["CFTime", "CommonDataModel", "DataStructures", "Dates", "DiskArrays", "NetCDF_jll", "NetworkOptions", "Printf"]
git-tree-sha1 = "5eb7747d10437f5acb2675c1f865ffa0353f3f9c"
uuid = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
version = "0.14.15"

    [deps.NCDatasets.extensions]
    NCDatasetsMPIExt = "MPI"

    [deps.NCDatasets.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "e2c3bba08dd6dedfe17a17889131b885b8c082f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.27"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "8a36db9b934b0e72583e624abc8f3b3d60554f2c"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.1000.0+0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3287ec88df50429a934ebc6cf14606215e27b987"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.33+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9ac7c730c53b3b5d9a73fb900ac4b4fc263774db"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.9+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "6d6c0ca4824268c1a7dca1f4721c535ac63d9074"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.11+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "dd974aefe288ef2898733aecf40858dc86742d74"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.1"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.aws_c_auth_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_http_jll", "aws_c_sdkutils_jll"]
git-tree-sha1 = "8cab83c96af80a1be968251ce1a0548a7545484d"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "0.9.6+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "22c0f42f4a1f0dc5dcfa8fd267c4ac407c455e7a"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "0.9.13+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a759cb9bf456ad792cc7898a81ae333cce9ef02a"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "0.12.6+0"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "7910c72f45f44afd297c39fe43b99c56d5ed22ec"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "0.3.2+0"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "e358d5a001ef7afbd4f8c5225322512819cda2f2"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "0.10.13+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "7e481d474b2087ee8bbf55b81bf9119f21e396d9"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "0.26.3+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "3e9917ab25114feba657e71be41cad068b9f6595"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "0.11.5+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "c43dfba2c1ab9ea9f02f2c80e86fa16f6460244e"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "0.2.4+1"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "2570c8e23f4771a087b12a47edcaaa670ac05a01"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "0.2.10+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1411bc34c180946d3cef591de1384012afa6edee"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.6+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "a8083ee0737c243c8f40a4ba86a0956997facb73"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "0.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.s2n_tls_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6b99e06a3863de281da6ff0e193a5b3706349054"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─d03ddc18-c35c-4fd8-9f92-e499c1c345c4
# ╟─a51c45f4-1f89-45e8-9967-b5b4c3bc6835
# ╠═6910d5fb-0522-499b-8210-d09950db41ad
# ╠═30d4a0b5-4ab0-45fa-b7c3-adc3a15c2edb
# ╟─545ecb96-47c5-44f2-82ed-178441e4b0ee
# ╠═94c1b1bd-6bbb-447d-963c-7d37569209b3
# ╠═d5c1a5ac-248a-439c-a45a-56a053a1cfad
# ╠═6c8313b3-08ca-4a7a-b130-3678b9dfa234
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000001
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa00000002
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000003
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa00000004
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000005
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa00000006
# ╟─a2fed5c7-0243-4eea-b187-7f24d7feb845
# ╟─eaff9fa5-3072-4f2d-848b-6d6705f6b65e
# ╠═1e3a6b29-61f7-46f4-a2a8-32f449faaf27
# ╟─848b1f1c-5edd-47e0-a1a3-a286c8dda937
# ╟─ce8f4301-cd81-4941-a57a-696c1b2806c3
# ╠═cc5ea7d9-b7cc-41a9-b73d-08e76a58c99b
# ╠═0087a2a7-d84e-4252-9a72-be059d26099e
# ╟─f6088bba-972b-4e10-9a3e-df140b6a768d
# ╟─f512c08c-b1f4-4104-824f-f5e0c2506f2f
# ╠═52acae2e-4dfc-4394-9404-546addc5a228
# ╟─147cf9a2-7582-48d1-bd4d-194b9d184480
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000007
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa00000008
# ╟─92eb9810-102f-4dbd-b618-604121d19f3b
# ╟─30d4a0b5-bbbb-cccc-dddd-100000000001
# ╠═369b550f-b097-4470-ae92-e601394f5db0
# ╟─30d4a0b5-bbbb-cccc-dddd-300000000001
# ╠═30d4a0b5-bbbb-cccc-dddd-300000000002
# ╟─992df9c2-7f00-433d-a4c8-26d02210ea6b
# ╠═90f298e4-6d17-4a81-b83c-169da804ff3b
# ╟─befa247c-0549-47d6-a30b-6a6fbad9ce3e
# ╟─506a5a59-29a5-447c-b0f9-1afe70b75251
# ╠═880a564f-d0a4-47b0-8fc1-a6f83e714112
# ╟─36b45df3-bb5c-4739-a21c-7f06d07ff491
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000009
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa0000000a
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa0000000b
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa0000000c
# ╟─c5eb626e-d6ff-473a-8adc-92ba2cdb10b1
# ╠═1e58cbed-8302-4c65-a79c-5ea3fabc1c1c
# ╟─ba48d31e-046a-48d5-a9e4-ac4117dd4855
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa0000000d
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa0000000e
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa0000000f
# ╠═30d4a0b5-bbbb-cccc-dddd-aaaa00000010
# ╟─30d4a0b5-bbbb-cccc-dddd-500000000001
# ╠═30d4a0b5-bbbb-cccc-dddd-500000000002
# ╟─30d4a0b5-bbbb-cccc-dddd-500000000003
# ╠═30d4a0b5-bbbb-cccc-dddd-500000000004
# ╟─30d4a0b5-bbbb-cccc-dddd-500000000005
# ╠═30d4a0b5-bbbb-cccc-dddd-500000000006
# ╟─30d4a0b5-bbbb-cccc-dddd-500000000007
# ╟─a33add4b-3acc-48c1-bc0d-97c814e3b39f
# ╠═dddbef5c-830b-4c67-816d-3e0d21e2288b
# ╠═843da5fd-0fb0-499a-8807-e2a09bb2d20c
# ╟─67db4c45-9a70-4304-b9d9-c05ec77537ff
# ╟─30d4a0b5-bbbb-cccc-dddd-400000000001
# ╠═30d4a0b5-bbbb-cccc-dddd-400000000002
# ╠═30d4a0b5-bbbb-cccc-dddd-400000000003
# ╟─30d4a0b5-bbbb-cccc-dddd-400000000004
# ╟─88921a71-c8a1-46dd-8d51-fb38d78d6b24
# ╠═7e12b7a2-b22a-4cb2-80cd-566b17454b2d
# ╠═1a1cfdf1-070e-47ab-9647-2f97f2fa1625
# ╟─c38ddd37-0dfc-43fa-bf05-b6e5cea6b9e2
# ╟─7a9ae0e7-0382-4f81-aab0-b8b42bbab1e8
# ╠═6a917333-1270-4ead-93c9-7bba4fc1b6fe
# ╠═1706f3c3-254d-408e-b635-739f9313185a
# ╟─00a1d8c0-29e1-48f2-bb0c-c4902f9fd77c
# ╟─30d4a0b5-bbbb-cccc-dddd-200000000001
# ╠═30d4a0b5-bbbb-cccc-dddd-200000000002
# ╠═30d4a0b5-bbbb-cccc-dddd-200000000003
# ╟─30d4a0b5-bbbb-cccc-dddd-200000000004
# ╟─5239ef9c-6555-433a-a749-79cf136aff0e
# ╟─30d4a0b5-bbbb-cccc-dddd-aaaa00000020
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
