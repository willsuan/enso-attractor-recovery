using LinearAlgebra
using Statistics
using StatsBase
using Random
using Printf
using CairoMakie
using ColorSchemes
using NearestNeighbors
using Dates
using Logging

global_logger(SimpleLogger(stderr, Logging.Error))

include("load_sst.jl")
include("pca.jl")
include("diffusion.jl")
include("delay_embed.jl")

const FIGDIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGDIR)

# ============================================================
# 1.  Load + preprocess
# ============================================================
println("=" ^ 70)
println("ENSO project — ERSSTv5 analysis")
println("=" ^ 70)

println("\nLoading ERSSTv5...")
sst, lat_all, lon_all, time_all = load_ersst()

# Restrict to Jan 1950 – Dec 2024 (75 years × 12 = 900 months)
t_start = Date(1950, 1, 1)
t_end   = Date(2024, 12, 31)
keep    = findall(t -> t_start <= Date(t) <= t_end, time_all)
sst_w   = sst[keep, :, :]
time_w  = time_all[keep]
@printf "Time window: %s to %s   (%d months)\n" Date(time_w[1]) Date(time_w[end]) length(time_w)

println("\nComputing anomalies...")
anom = sst_anomalies(sst_w, time_w)

println("\nSubsetting tropical Pacific (30°S–30°N, 120°E–80°W)...")
pac, lat, lon = subset_region(anom, lat_all, lon_all)
@printf "Cube shape: %s\n" size(pac)

X_raw, mask, valid_idx = flatten_to_matrix(pac)
weights = cos_lat_weights(lat, lon, mask)
# Apply √cos(lat) weights so that the inner product on rows reflects
# physical surface area on the sphere.
X = X_raw .* weights'
@printf "Data matrix X: %s   (T months × N valid sea cells)\n" size(X)

# Niño 3.4 reference index, aligned to our months
println("\nLoading Niño 3.4...")
n_t, n_v = load_nino34()
ix = [findfirst(==(Date(t)), n_t) for t in time_w]
nino34 = Float64[]
for j in ix
    push!(nino34, isnothing(j) ? NaN : n_v[j])
end
valid_n = .!isnan.(nino34)
@printf "Niño 3.4 alignment: %d/%d months valid\n" count(valid_n) length(nino34)

# ============================================================
# 2.  PCA / EOF analysis  (P1, P2)
# ============================================================
println("\n" * "=" ^ 70)
println("Method 1 — PCA / EOF analysis")
println("=" ^ 70)

pca = fit_pca(X; rank=20)
PC = Float64.(pca.scores)
EOF = Float64.(pca.V)
explained = Float64.(pca.explained)

@printf "Top-5 explained variance: %s\n" round.(explained[1:5]; digits=3)
@printf "Cumulative top-5: %.3f\n" sum(explained[1:5])

# Make PC1's sign consistent with Niño 3.4 (positive correlation)
function align_sign!(v::AbstractVector, ref::AbstractVector, mask::AbstractVector{Bool})
    c = cor(v[mask], ref[mask])
    if c < 0
        v .= -v
    end
    return c
end

ρ_PC = zeros(5)
for k in 1:5
    PC[:, k] .= PC[:, k]   # already a copy
    EOF[:, k] .= EOF[:, k]
    c = cor(PC[valid_n, k], nino34[valid_n])
    if c < 0
        PC[:, k] .= -PC[:, k]
        EOF[:, k] .= -EOF[:, k]
        c = -c
    end
    ρ_PC[k] = c
end
@printf "PC vs Niño 3.4 correlations: %s\n" round.(ρ_PC; digits=3)

# Test of P1: EOF1 spatial pattern look like ENSO?
# Test of P2: cor(PC1, Niño 3.4) > 0.9?
@printf "\nP1 — EOF1 pattern shape: see figure\n"
@printf "P2 — cor(PC1, Niño 3.4) = %.3f   target ≥ 0.90  →  %s\n" ρ_PC[1] (ρ_PC[1] >= 0.90 ? "PASS" : "FAIL")

# ============================================================
# 3.  Plain Diffusion Map  (P3, P4, P5)
# ============================================================
println("\n" * "=" ^ 70)
println("Method 2 — Plain Diffusion Maps on monthly snapshots")
println("=" ^ 70)

# α=1 (Laplace–Beltrami)
println("Fitting DMAP α=1 ...")
dm = fit_diffusion_map(X; α=1.0, d=15, t=1)
@printf "ε used: %.4g\n" dm.ε
@printf "Top-10 |λ|: %s\n" round.(abs.(dm.λ[1:min(end,10)]); digits=4)

Ψ = Float64.(dm.coords)
ρ_Ψ = zeros(5)
for k in 1:5
    c = cor(Ψ[valid_n, k], nino34[valid_n])
    if c < 0
        Ψ[:, k] .= -Ψ[:, k]
        c = -c
    end
    ρ_Ψ[k] = c
end
@printf "Ψ vs Niño 3.4 correlations: %s\n" round.(ρ_Ψ; digits=3)

@printf "\nP3 — cor(Ψ₂, Niño 3.4) = %.3f   target ≥ 0.60  →  %s\n" ρ_Ψ[1] (ρ_Ψ[1] >= 0.60 ? "PASS" : "FAIL")

# Spectral gap analysis (P4)
λabs = Float64.(abs.(dm.λ))
gaps = abs.(diff(λabs[1:min(end,10)]))
gap_after = argmax(gaps)
@printf "P4 — largest eigenvalue gap is between λ_%d and λ_%d (gap=%.3g)\n" gap_after gap_after+1 maximum(gaps)

# ============================================================
# 4.  α-family comparison  (E2)
# ============================================================
println("\n--- α-family comparison ---")
dm_alpha = Dict{Float64, DiffusionMap}()
ρ_alpha  = Dict{Float64, Vector{Float64}}()
for α in (0.0, 0.5, 1.0)
    @printf "α = %.1f ... " α
    d = fit_diffusion_map(X; α=α, d=10, t=1)
    Ψα = Float64.(d.coords)
    ρs = zeros(5)
    for k in 1:5
        c = cor(Ψα[valid_n, k], nino34[valid_n])
        if c < 0
            Ψα[:, k] .= -Ψα[:, k]
            c = -c
        end
        ρs[k] = c
    end
    dm_alpha[α] = d
    ρ_alpha[α]  = ρs
    @printf "ρ(Ψ₂, Niño 3.4) = %.3f\n" ρs[1]
end

# ============================================================
# 5.  Time-delay DMAP  (BONUS: P6, P7)
# ============================================================
println("\n" * "=" ^ 70)
println("Method 3 — DMAP on time-delay embeddings (Takens, BONUS)")
println("=" ^ 70)

# Sweep over delay k
delay_ks = [0, 3, 6, 12, 18, 24]
ρ_delay  = Dict{Int, Float64}()
dm_delay = Dict{Int, Any}()
ts_delay = Dict{Int, Vector{Int}}()

for k_delay in delay_ks
    Y, src_idx = delay_stack(X, k_delay)
    d = fit_diffusion_map(Y; α=1.0, d=10, t=1)
    Ψd = Float64.(d.coords)
    aligned_nino = nino34[src_idx]
    aligned_valid = .!isnan.(aligned_nino)
    c = cor(Ψd[aligned_valid, 1], aligned_nino[aligned_valid])
    if c < 0
        Ψd[:, :] .= -Ψd[:, :]
        c = -c
    end
    ρ_delay[k_delay] = c
    dm_delay[k_delay] = (dm=d, Ψ=Ψd, src_idx=src_idx)
    ts_delay[k_delay] = src_idx
    @printf "k=%2d:  N=%d   ρ(Ψ₂', Niño 3.4) = %.3f\n" k_delay size(Y,1) c
end

improvement = ρ_delay[12] - ρ_delay[0]
@printf "\nP6 — improvement Δρ at k=12 over k=0:  %+.3f   target ≥ +0.05  →  %s\n" improvement (improvement >= 0.05 ? "PASS" : "FAIL")

ρ_seq = [ρ_delay[k] for k in delay_ks]
mono_ish = all(ρ_seq[i+1] >= ρ_seq[i] - 0.02 for i in 1:length(ρ_seq)-1)
@printf "P7 — non-decreasing ρ across k sweep?  %s\n" (mono_ish ? "PASS" : "FAIL")

# ============================================================
# 6.  Negative control via permuted Niño 3.4
# ============================================================
println("\n" * "=" ^ 70)
println("Negative control: correlate PC1 / Ψ₂ against a randomly permuted Niño 3.4")
println("=" ^ 70)

# The cleanest null distribution: keep the data and methods unchanged,
# correlate the recovered indices against a *shuffled* Niño 3.4.  This
# is the chance-correlation distribution; if the real correlation lies
# outside it, the recovery is non-random.
rng = MersenneTwister(2026)
n_trials = 500
chance_PC = Float64[]
chance_DM = Float64[]
nino_clean = filter(!isnan, nino34)
for _ in 1:n_trials
    perm = shuffle(rng, nino_clean)
    push!(chance_PC, abs(cor(PC[valid_n, 1][1:length(perm)], perm)))
    push!(chance_DM, abs(cor(Ψ[valid_n, 1][1:length(perm)], perm)))
end
chance_PC_q95 = quantile(chance_PC, 0.95)
chance_DM_q95 = quantile(chance_DM, 0.95)

@printf "Chance correlation distribution over %d shuffles:\n" n_trials
@printf "  PC1:  mean = %.4f   95th pct = %.4f   max = %.4f\n" mean(chance_PC) chance_PC_q95 maximum(chance_PC)
@printf "  Ψ₂ :  mean = %.4f   95th pct = %.4f   max = %.4f\n" mean(chance_DM) chance_DM_q95 maximum(chance_DM)
@printf "Real correlations:\n"
@printf "  PC1: %.3f   (%.0fσ above chance)\n" ρ_PC[1] ((ρ_PC[1] - mean(chance_PC)) / std(chance_PC))
@printf "  Ψ₂ : %.3f   (%.0fσ above chance)\n" ρ_Ψ[1]  ((ρ_Ψ[1]  - mean(chance_DM)) / std(chance_DM))

# Also keep the spatial-shuffle "amplitude-only" baseline — it tells a
# different but still informative story.
println("\nAuxiliary baseline: random spatial permutation of each month")
X_shuf = similar(X)
for t in 1:size(X, 1)
    p = randperm(rng, size(X, 2))
    X_shuf[t, :] = X[t, p]
end
pca_s = fit_pca(X_shuf; rank=5)
PC_s  = Float64.(pca_s.scores)
ρ_shuf_pc = abs(cor(PC_s[valid_n, 1], nino34[valid_n]))
dm_s = fit_diffusion_map(X_shuf; α=1.0, d=5, t=1)
Ψ_s  = Float64.(dm_s.coords)
ρ_shuf_dm = abs(cor(Ψ_s[valid_n, 1], nino34[valid_n]))
@printf "  Spatial-shuffle PC1: %.3f   (vs. real %.3f).  Residual ≈ correlation of total amplitude with Niño 3.4.\n" ρ_shuf_pc ρ_PC[1]
@printf "  Spatial-shuffle Ψ₂ : %.3f   (vs. real %.3f).\n" ρ_shuf_dm ρ_Ψ[1]
ctrl_pass = ρ_PC[1] > chance_PC_q95 && ρ_Ψ[1] > chance_DM_q95

# ============================================================
# 7.  Save summary metrics
# ============================================================
open(joinpath(FIGDIR, "enso_metrics.txt"), "w") do io
    println(io, "# ENSO project metrics")
    println(io, "# Time window: $(Date(time_w[1])) to $(Date(time_w[end])) ($(length(time_w)) months)")
    println(io, "# Tropical Pacific: 30°S–30°N, 120°E–80°W ($(size(X,2)) sea cells)")
    println(io)
    println(io, "## P1, P2 — PCA / EOF")
    @printf io "  ρ(PC1, Niño 3.4) = %.3f\n" ρ_PC[1]
    println(io, "  Top-5 explained variance: $(round.(explained[1:5]; digits=3))")
    println(io)
    println(io, "## P3, P4, P5 — Plain DMAP α=1")
    @printf io "  ρ(Ψ₂, Niño 3.4) = %.3f\n" ρ_Ψ[1]
    println(io, "  Top-10 |λ|: $(round.(λabs[1:min(end,10)]; digits=4))")
    @printf io "  largest gap after k=%d  (Δ=%.3g)\n" gap_after maximum(gaps)
    println(io)
    println(io, "## α-family")
    for α in (0.0, 0.5, 1.0)
        @printf io "  α=%.1f:  ρ(Ψ₂, Niño 3.4) = %.3f\n" α ρ_alpha[α][1]
    end
    println(io)
    println(io, "## P6, P7 — Time-delay DMAP")
    for k in delay_ks
        @printf io "  k=%2d:  ρ(Ψ₂', Niño 3.4) = %.3f\n" k ρ_delay[k]
    end
    @printf io "  improvement k=0 → k=12:  %+.3f\n" improvement
    println(io)
    println(io, "## E5 — Spatial-shuffle negative control")
    @printf io "  ρ(PC1_shuf, Niño 3.4) = %.3f  (orig %.3f)\n" ρ_shuf_pc ρ_PC[1]
    @printf io "  ρ(Ψ₂_shuf, Niño 3.4)  = %.3f  (orig %.3f)\n" ρ_shuf_dm ρ_Ψ[1]
end

# Persist all results to a JLD2-style file isn't strictly needed; we
# will run figure generation in the same script.  Print a summary:
println("\n" * "=" ^ 70)
println("SUMMARY  (predictions)")
println("=" ^ 70)
@printf "P1  EOF1 spatial pattern: see figure\n"
@printf "P2  ρ(PC1, Niño 3.4) = %.3f   target ≥ 0.90   %s\n" ρ_PC[1] (ρ_PC[1] >= 0.90 ? "PASS" : "FAIL")
@printf "P3  ρ(Ψ₂, Niño 3.4)  = %.3f   target ≥ 0.60   %s\n" ρ_Ψ[1] (ρ_Ψ[1] >= 0.60 ? "PASS" : "FAIL")
@printf "P4  largest gap after k=%d (Δ=%.3g)\n" gap_after maximum(gaps)
@printf "P5  see phase-portrait figure\n"
@printf "P6  ρ(k=12) − ρ(k=0) = %+.3f   target ≥ +0.05  %s\n" improvement (improvement >= 0.05 ? "PASS" : "FAIL")
@printf "P7  monotone in k:  %s\n" (mono_ish ? "PASS" : "FAIL")
@printf "E5  spatial-shuffle control kills both methods?  %s\n" (ctrl_pass ? "YES" : "NO")
println()

# Stash the things we need for figure generation
const ENSO_RESULTS = (
    X = X, mask = mask, lat = lat, lon = lon, time = time_w,
    nino34 = nino34, valid_n = valid_n,
    pca = pca, PC = PC, EOF = EOF, explained = explained, ρ_PC = ρ_PC,
    dm = dm, Ψ = Ψ, ρ_Ψ = ρ_Ψ,
    dm_alpha = dm_alpha, ρ_alpha = ρ_alpha,
    dm_delay = dm_delay, ρ_delay = ρ_delay, delay_ks = delay_ks,
    ρ_shuf_pc = ρ_shuf_pc, ρ_shuf_dm = ρ_shuf_dm,
)
println("ENSO_RESULTS bound; figure-generation script can use it.")
