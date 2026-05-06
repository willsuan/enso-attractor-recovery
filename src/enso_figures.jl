using LinearAlgebra
using Statistics
using StatsBase
using Random
using Printf
using CairoMakie
using ColorSchemes
using Dates
using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

include("load_sst.jl")
include("pca.jl")
include("diffusion.jl")
include("delay_embed.jl")

const FIGDIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGDIR)

# Clear the old hyperspectral figures so the report figures/ dir is fresh
for f in readdir(FIGDIR; join=true)
    if startswith(basename(f), "fig") || basename(f) == "metrics.txt"
        rm(f; force=true)
    end
end

# Reload data + run analysis (compact version of enso_analysis.jl)
println("Loading + preprocessing...")
sst, lat_all, lon_all, time_all = load_ersst()
keep = findall(t -> Date(1950,1,1) <= Date(t) <= Date(2024,12,31), time_all)
sst_w  = sst[keep, :, :]
time_w = time_all[keep]
anom = sst_anomalies(sst_w, time_w)
pac, lat, lon = subset_region(anom, lat_all, lon_all)
X_raw, mask, _ = flatten_to_matrix(pac)
weights = cos_lat_weights(lat, lon, mask)
X = X_raw .* reshape(weights, 1, :)

n_t, n_v = load_nino34()
ix = [findfirst(==(Date(t)), n_t) for t in time_w]
nino34 = Float64[isnothing(j) ? NaN : n_v[j] for j in ix]
valid_n = .!isnan.(nino34)

# PCA
println("Fitting PCA...")
pca = fit_pca(X; rank=20)
PC = Float64.(pca.scores)
EOF = Float64.(pca.V)
explained = Float64.(pca.explained)
for k in 1:5
    if cor(PC[valid_n, k], nino34[valid_n]) < 0
        PC[:, k] .= -PC[:, k]
        EOF[:, k] .= -EOF[:, k]
    end
end
ρ_PC = [cor(PC[valid_n, k], nino34[valid_n]) for k in 1:5]
@printf "PC1 vs N3.4: %.3f\n" ρ_PC[1]

# DMAP α-family
println("Fitting DMAP α-family...")
dm_alpha = Dict{Float64, Any}()
ρ_alpha  = Dict{Float64, Vector{Float64}}()
for α in (0.0, 0.5, 1.0)
    d = fit_diffusion_map(X; α=α, d=10, t=1)
    Ψ = Float64.(d.coords)
    ρs = zeros(5)
    for k in 1:5
        c = cor(Ψ[valid_n, k], nino34[valid_n])
        if c < 0
            Ψ[:, k] .= -Ψ[:, k]
            c = -c
        end
        ρs[k] = c
    end
    dm_alpha[α] = (dm=d, Ψ=Ψ)
    ρ_alpha[α]  = ρs
    @printf "  α=%.1f: ρ(Ψ₂, N3.4)=%.3f\n" α ρs[1]
end

# Use α=1 as primary
dm = dm_alpha[1.0].dm
Ψ  = dm_alpha[1.0].Ψ
ρ_Ψ = ρ_alpha[1.0]

# Time-delay sweep
println("Time-delay sweep (raw spatial)...")
delay_ks = [0, 3, 6, 9, 12, 18, 24]
ρ_delay  = Dict{Int, Float64}()       # leading Ψ₂' correlation
ρ_delay_max = Dict{Int, Float64}()    # max over Ψ₂'..Ψ₆'
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
    ρ_delay[kd] = cs[1]
    ρ_delay_max[kd] = maximum(cs)
    @printf "  k=%2d:  ρ(Ψ₂')=%.3f   max ρ(Ψ_k')=%.3f\n" kd cs[1] maximum(cs)
end

# Time-delay in PC space
println("Time-delay sweep (PC-space, top 10)...")
ρ_delay_pc  = Dict{Int, Float64}()
ρ_delay_pc_max = Dict{Int, Float64}()
for kd in delay_ks
    if kd == 0
        Y = PC[:, 1:10]
        src = collect(1:size(Y,1))
    else
        Y, src = delay_stack(PC[:, 1:10], kd)
    end
    d = fit_diffusion_map(Y; α=1.0, d=6, t=1)
    Ψd = Float64.(d.coords)
    aligned = nino34[src]
    valid = .!isnan.(aligned)
    cs = [abs(cor(Ψd[valid, k], aligned[valid])) for k in 1:5]
    ρ_delay_pc[kd] = cs[1]
    ρ_delay_pc_max[kd] = maximum(cs)
    @printf "  k=%2d:  ρ(Ψ₂')=%.3f   max ρ(Ψ_k')=%.3f\n" kd cs[1] maximum(cs)
end

# Negative control
println("Permutation null distribution (n=500)...")
rng = MersenneTwister(2026)
n_clean_idx = findall(valid_n)
n_clean = nino34[valid_n]
chance_PC = Float64[]
chance_DM = Float64[]
for _ in 1:500
    p = shuffle(rng, n_clean)
    push!(chance_PC, abs(cor(PC[valid_n, 1], p)))
    push!(chance_DM, abs(cor(Ψ[valid_n, 1], p)))
end

# 1.  EOF1 spatial pattern + PC1 time series  (P1, P2)
println("\nFig 1: EOF1 spatial pattern")

function eof_grid(eof_vec, mask, weights)
    # Undo the cos-lat weighting and reshape onto the lat/lon grid
    inv_w = 1.0 ./ weights
    eof_unw = eof_vec .* inv_w
    La, Lo = size(mask)
    G = fill(NaN32, La, Lo)
    flat_idx = findall(mask)
    @inbounds for (k, ci) in enumerate(flat_idx)
        G[ci] = eof_unw[k]
    end
    return G
end

let
    fig = Figure(size = (1100, 480), backgroundcolor=:white)
    ax = Axis(fig[1, 1];
        title  = "EOF 1, leading mode of tropical Pacific SST anomalies",
        xlabel = "longitude (°E)", ylabel = "latitude (°N)",
        aspect = 2.5)
    G = eof_grid(EOF[:, 1], mask, weights)
    # Diverging colormap; symmetric range
    vmax = maximum(abs, filter(!isnan, G))
    hm = heatmap!(ax, lon, lat, G';
        colormap = :RdBu_11, colorrange = (-vmax, vmax))
    Colorbar(fig[1, 2], hm; label="EOF 1 amplitude (a.u.)")

    # Niño 3.4 box overlay
    n34_lon = (190, 240)    # 170°W to 120°W in 0–360 convention
    n34_lat = (-5, 5)
    poly!(ax, Point2f[(n34_lon[1], n34_lat[1]), (n34_lon[2], n34_lat[1]),
                      (n34_lon[2], n34_lat[2]), (n34_lon[1], n34_lat[2])];
          color = (:black, 0), strokecolor = :black, strokewidth = 2,
          linestyle = :dash)
    text!(ax, n34_lon[1] + 5, n34_lat[2] + 1; text = "Niño 3.4 box",
          fontsize = 11, color = :black)

    save(joinpath(FIGDIR, "fig01_eof1.png"), fig)
end

# 2.  PC 1 vs Niño 3.4 time series  (P2)
println("Fig 2: PC1 vs Niño 3.4 overlay")
let
    fig = Figure(size = (1300, 380))
    ax = Axis(fig[1, 1];
        title  = "PC 1 of tropical Pacific SST anomalies vs. Niño 3.4 index   (ρ = $(round(ρ_PC[1]; digits=3)))",
        xlabel = "year", ylabel = "anomaly (Niño 3.4: °C)")

    years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]

    # Standardise PC1 to the same scale as Niño 3.4 for visual overlay
    s = std(nino34[valid_n]) / std(PC[:, 1])
    pc_scaled = PC[:, 1] .* s

    lines!(ax, years, nino34; color = (:steelblue, 0.85), label = "Niño 3.4 (NOAA)", linewidth = 1.5)
    lines!(ax, years, pc_scaled; color = (:crimson, 0.85), label = "PC 1 (rescaled)", linewidth = 1.2)
    hlines!(ax, [0]; color = :gray, linewidth = 0.6)

    # Mark the strongest historical El Niños
    big_events = [Date(1972, 12, 1), Date(1982, 12, 1), Date(1997, 12, 1),
                  Date(2015, 12, 1), Date(2023, 12, 1)]
    for d in big_events
        idx = findfirst(==(d), Date.(time_w))
        if !isnothing(idx)
            scatter!(ax, [years[idx]], [nino34[idx]];
                     color = :black, markersize = 10, marker = :star5)
        end
    end
    axislegend(ax; position = :rt, framevisible = false)

    save(joinpath(FIGDIR, "fig02_pc1_nino34.png"), fig)
end

# 3.  DMAP eigenvalues + α-family side-by-side scatter
println("Fig 3: DMAP eigenvalues + α-family scatters")
let
    fig = Figure(size = (1500, 700))

    # Eigenvalue spectra (top row, full width)
    ax = Axis(fig[1, 1:4]; title = "Diffusion-map eigenvalues |λₖ|, α-family",
        xlabel = "k", ylabel = "|λₖ|")
    cols = Dict(0.0 => :gray, 0.5 => :darkorange, 1.0 => :teal)
    for α in (0.0, 0.5, 1.0)
        λ = abs.(Float64.(dm_alpha[α].dm.λ))
        n = min(15, length(λ))
        scatterlines!(ax, 1:n, λ[1:n]; color = cols[α], markersize = 8,
                      label = "α = $α")
    end
    axislegend(ax; position = :rt, framevisible = false)

    # PC1 vs PC2 scatter, coloured by Niño 3.4
    ax2_1 = Axis(fig[2, 1]; title = "PCA: PC1 vs PC2",
        xlabel = "PC 1", ylabel = "PC 2")
    sct = scatter!(ax2_1, PC[:, 1], PC[:, 2]; color = nino34, colormap = :RdBu_11,
                   colorrange = (-2.5, 2.5), markersize = 4, alpha = 0.7)

    for (j, α) in enumerate((0.0, 0.5, 1.0))
        ax2 = Axis(fig[2, j + 1]; title = "DMAP α=$α: Ψ₂ vs Ψ₃",
            xlabel = "Ψ₂", ylabel = "Ψ₃")
        Ψα = dm_alpha[α].Ψ
        scatter!(ax2, Ψα[:, 1], Ψα[:, 2]; color = nino34, colormap = :RdBu_11,
                colorrange = (-2.5, 2.5), markersize = 4, alpha = 0.7)
    end

    Colorbar(fig[3, 1:4], sct; label = "Niño 3.4 (°C)",
             vertical = false, height = 14, width = Relative(0.5))

    rowsize!(fig.layout, 3, Fixed(40))

    save(joinpath(FIGDIR, "fig03_eigenvalues_alpha_scatter.png"), fig)
end

# 4.  Phase portrait (Ψ₂, Ψ₃) with monthly trajectory  (P5)
println("Fig 4: phase portrait + trajectory")
let
    fig = Figure(size = (1300, 500))

    # Standalone scatter coloured by Niño 3.4
    ax1 = Axis(fig[1, 1]; title = "Phase portrait Ψ₂ vs Ψ₃, all 900 months",
        xlabel = "Ψ₂", ylabel = "Ψ₃")
    sct = scatter!(ax1, Ψ[:, 1], Ψ[:, 2]; color = nino34, colormap = :RdBu_11,
                   colorrange = (-2.5, 2.5), markersize = 5, alpha = 0.7)
    Colorbar(fig[1, 2], sct; label = "Niño 3.4 (°C)")

    # Mark major El Niño months
    big_events = Dict(
        Date(1972, 12, 1) => "Dec 1972",
        Date(1982, 12, 1) => "Dec 1982",
        Date(1997, 12, 1) => "Dec 1997",
        Date(2015, 12, 1) => "Dec 2015",
        Date(2023, 12, 1) => "Dec 2023",
    )
    for (d, lbl) in big_events
        idx = findfirst(==(d), Date.(time_w))
        if !isnothing(idx)
            scatter!(ax1, [Ψ[idx, 1]], [Ψ[idx, 2]];
                     color = :black, markersize = 14, marker = :star5)
            text!(ax1, Ψ[idx, 1] + 0.005, Ψ[idx, 2]; text = " $lbl",
                  fontsize = 10, color = :black)
        end
    end

    # Trajectory: connect consecutive months for one strong El Niño cycle
    ax2 = Axis(fig[1, 3]; title = "Trajectory through one ENSO cycle (Jan 1996–Dec 1999)",
        xlabel = "Ψ₂", ylabel = "Ψ₃")
    cycle_start = findfirst(==(Date(1996, 1, 1)), Date.(time_w))
    cycle_end   = findfirst(==(Date(1999, 12, 1)), Date.(time_w))
    seg = cycle_start:cycle_end
    months_seg = 1:length(seg)
    sct2 = scatter!(ax2, Ψ[seg, 1], Ψ[seg, 2]; color = nino34[seg],
                    colormap = :RdBu_11, colorrange = (-2.5, 2.5), markersize = 8)
    lines!(ax2,  Ψ[seg, 1], Ψ[seg, 2]; color = (:black, 0.4), linewidth = 1.0)
    # arrowheads at every 6 months
    for i in 6:6:length(seg)-1
        arrows!(ax2, [Ψ[seg[i], 1]], [Ψ[seg[i], 2]],
                [Ψ[seg[i+1], 1] - Ψ[seg[i], 1]], [Ψ[seg[i+1], 2] - Ψ[seg[i], 2]];
                color = :black, linewidth = 1.5)
    end

    save(joinpath(FIGDIR, "fig04_phase_portrait.png"), fig)
end

# 5.  α-family Ψ₂ correlations and full correlation table  (P3)
println("Fig 5: PC and Ψ correlations across modes and α")
let
    fig = Figure(size = (1100, 400))
    ax1 = Axis(fig[1, 1]; title = "Correlation with Niño 3.4 by mode index",
        xlabel = "mode k (PC_k or Ψ_{k+1})", ylabel = "|ρ|")

    scatterlines!(ax1, 1:5, ρ_PC[1:5]; color = :crimson, markersize = 10,
                  label = "PCA (PC_k)")
    cols = Dict(0.0 => :gray, 0.5 => :darkorange, 1.0 => :teal)
    for α in (0.0, 0.5, 1.0)
        scatterlines!(ax1, 1:5, ρ_alpha[α][1:5]; color = cols[α], markersize = 8,
                      label = "DMAP α=$α (Ψ_{k+1})")
    end
    axislegend(ax1; position = :rt, framevisible = false)

    # Bar chart: leading-mode correlation by method
    ax2 = Axis(fig[1, 2]; title = "Leading-mode correlation with Niño 3.4",
        xticks = (1:4, ["PCA", "DMAP α=0", "DMAP α=½", "DMAP α=1"]),
        ylabel = "|ρ|")
    barplot!(ax2, [1, 2, 3, 4],
             [ρ_PC[1], ρ_alpha[0.0][1], ρ_alpha[0.5][1], ρ_alpha[1.0][1]];
             color = [:crimson, :gray, :darkorange, :teal])
    ylims!(ax2, 0, 1.0)

    save(joinpath(FIGDIR, "fig05_correlations.png"), fig)
end

# 6.  Time-delay sweep (method-stacking robustness check)
println("Fig 6: time-delay sensitivity (raw vs PC space)")
let
    fig = Figure(size = (1100, 380))
    ax = Axis(fig[1, 1];
        title = "Time-delay embedding: leading-mode correlation",
        xlabel = "delay k (months)", ylabel = "|ρ(Ψ₂', Niño 3.4)|")
    scatterlines!(ax, delay_ks, [ρ_delay[k] for k in delay_ks];
        color = :crimson, markersize = 10, label = "raw spatial")
    scatterlines!(ax, delay_ks, [ρ_delay_pc[k] for k in delay_ks];
        color = :teal, markersize = 10, label = "PC-space (top 10)")
    hlines!(ax, [ρ_PC[1]]; color = :gray, linestyle = :dash, label = "PCA baseline")
    axislegend(ax; position = :rb, framevisible = false)

    ax2 = Axis(fig[1, 2];
        title = "Same, but max ρ over Ψ₂'..Ψ₆'",
        xlabel = "delay k (months)", ylabel = "max_k |ρ(Ψ_k', Niño 3.4)|")
    scatterlines!(ax2, delay_ks, [ρ_delay_max[k] for k in delay_ks];
        color = :crimson, markersize = 10, label = "raw spatial")
    scatterlines!(ax2, delay_ks, [ρ_delay_pc_max[k] for k in delay_ks];
        color = :teal, markersize = 10, label = "PC-space (top 10)")
    hlines!(ax2, [ρ_PC[1]]; color = :gray, linestyle = :dash, label = "PCA baseline")
    axislegend(ax2; position = :rb, framevisible = false)

    save(joinpath(FIGDIR, "fig06_delay_sweep.png"), fig)
end

# 7.  Negative control histogram
println("Fig 7: permutation null vs. real correlation")
let
    fig = Figure(size = (1100, 360))
    ax = Axis(fig[1, 1];
        title = "Null distribution of |ρ(method, shuffled Niño 3.4)|, n=500",
        xlabel = "|ρ|", ylabel = "count")
    hist!(ax, chance_PC; bins = 30, color = (:crimson, 0.5),
          strokewidth = 0, label = "shuffled (PCA)")
    hist!(ax, chance_DM; bins = 30, color = (:teal, 0.5),
          strokewidth = 0, label = "shuffled (DMAP)")
    vlines!(ax, [ρ_PC[1]]; color = :crimson, linewidth = 3,
            label = "real PC1 = $(round(ρ_PC[1]; digits=3))")
    vlines!(ax, [ρ_Ψ[1]];  color = :teal,    linewidth = 3,
            label = "real Ψ₂ = $(round(ρ_Ψ[1]; digits=3))")
    axislegend(ax; position = :rt, framevisible = false)

    # Sigma annotation
    σ_PC = (ρ_PC[1] - mean(chance_PC)) / std(chance_PC)
    σ_DM = (ρ_Ψ[1]  - mean(chance_DM)) / std(chance_DM)
    text!(ax, 0.5, 0.95;  text = "PC1 is $(round(Int, σ_PC))σ above null",
        space = :relative, color = :crimson, fontsize = 12)
    text!(ax, 0.5, 0.88; text = "Ψ₂ is $(round(Int, σ_DM))σ above null",
        space = :relative, color = :teal, fontsize = 12)

    save(joinpath(FIGDIR, "fig07_null_distribution.png"), fig)
end

# 8.  Bandwidth (ε) sensitivity
println("Fig 8: bandwidth sensitivity")
let
    # Compute Ψ₂ vs Niño 3.4 correlation across a range of ε scales
    # relative to the median bandwidth.
    # Use the *same* median ε computed from the kernel, scaled.
    # We re-run DMAP at α=1 for each scale.
    base_ε = dm.ε
    scales = 10.0 .^ range(-1, 1.5; length=11)
    corrs  = Float64[]
    for s in scales
        d = fit_diffusion_map(X; α=1.0, ε=base_ε * s, d=5, t=1)
        Ψi = Float64.(d.coords)
        c = abs(cor(Ψi[valid_n, 1], nino34[valid_n]))
        push!(corrs, c)
    end

    fig = Figure(size = (1100, 360))
    ax = Axis(fig[1, 1];
        title = "Bandwidth sensitivity: |ρ(Ψ₂, Niño 3.4)| vs ε / ε_median",
        xlabel = "ε / ε_median", ylabel = "|ρ|", xscale = log10)
    scatterlines!(ax, scales, corrs; color = :teal, markersize = 10)
    hlines!(ax, [ρ_PC[1]]; color = :crimson, linestyle = :dash,
        label = "PCA baseline ρ=$(round(ρ_PC[1]; digits=3))")
    axislegend(ax; position = :lb, framevisible = false)

    save(joinpath(FIGDIR, "fig08_bandwidth.png"), fig)
end

# Save metrics to text file
open(joinpath(FIGDIR, "enso_metrics.txt"), "w") do io
    @printf io "# ENSO project metrics  (ERSSTv5, 1950–2024, tropical Pacific)\n\n"
    @printf io "## P1, P2, PCA / EOF\n"
    @printf io "ρ(PC1, Niño 3.4) = %.3f\n" ρ_PC[1]
    @printf io "Top-5 explained variance: %s\n" round.(explained[1:5]; digits=3)
    @printf io "\n"
    @printf io "## P3, P4, P5, Plain DMAP α=1\n"
    @printf io "ρ(Ψ₂, Niño 3.4) = %.3f\n" ρ_Ψ[1]
    @printf io "ρ(Ψ_k, Niño 3.4) for k=1..5: %s\n" round.(ρ_Ψ; digits=3)
    @printf io "\n"
    @printf io "## α-family\n"
    for α in (0.0, 0.5, 1.0)
        @printf io "α=%.1f:  ρ(Ψ₂)=%.3f\n" α ρ_alpha[α][1]
    end
    @printf io "\n"
    @printf io "## P6, P7, time-delay embedding (NEGATIVE RESULT)\n"
    for k in delay_ks
        @printf io "k=%2d:  ρ(Ψ₂') raw=%.3f  PC=%.3f  max_raw=%.3f  max_PC=%.3f\n" k ρ_delay[k] ρ_delay_pc[k] ρ_delay_max[k] ρ_delay_pc_max[k]
    end
    @printf io "Δρ(k=12 vs k=0) raw  = %+.3f   (target ≥ +0.05  → FAIL)\n" (ρ_delay[12] - ρ_delay[0])
    @printf io "Δρ(k=12 vs k=0) PC   = %+.3f   (target ≥ +0.05  → FAIL)\n" (ρ_delay_pc[12] - ρ_delay_pc[0])
    @printf io "\n"
    @printf io "## Negative control\n"
    @printf io "Permutation null (n=500):\n"
    @printf io "  PC: mean=%.4f, 95th=%.4f, max=%.4f\n" mean(chance_PC) quantile(chance_PC, 0.95) maximum(chance_PC)
    @printf io "  DM: mean=%.4f, 95th=%.4f, max=%.4f\n" mean(chance_DM) quantile(chance_DM, 0.95) maximum(chance_DM)
    @printf io "Real correlations are %.0fσ (PCA) and %.0fσ (DMAP) above the null distribution\n" ((ρ_PC[1] - mean(chance_PC)) / std(chance_PC)) ((ρ_Ψ[1] - mean(chance_DM)) / std(chance_DM))
end

# 9.  Wavelet spectrogram of PC 1 (Lecture 15 + 27 connection)
println("\nFig 9: PC 1 wavelet spectrogram")
include("spectrogram.jl")
let
    dt_yr = 1/12
    W_pc, scales_pc, periods_pc, coi_pc = morlet_cwt(PC[:, 1], dt_yr;
                                                      dj = 1/16, s0 = 2*dt_yr)
    P_pc = abs2.(W_pc)
    σ² = var(PC[:, 1])
    P_norm = P_pc ./ σ²

    years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]

    fig = Figure(size = (1300, 540))
    ax1 = Axis(fig[1, 1];
        title  = "PC 1 of tropical Pacific SST anomalies vs Niño 3.4   (ρ = $(round(ρ_PC[1]; digits=3)))",
        xlabel = "year", ylabel = "anomaly")
    s = std(nino34[valid_n]) / std(PC[:, 1])
    lines!(ax1, years, nino34;        color = (:steelblue, 0.85), label = "Niño 3.4")
    lines!(ax1, years, PC[:, 1] .* s; color = (:crimson, 0.85),    label = "PC 1 (rescaled)")
    hlines!(ax1, [0]; color = :gray, linewidth = 0.5)
    axislegend(ax1; position = :rt, framevisible = false)

    ax2 = Axis(fig[2, 1];
        title  = "Morlet wavelet power spectrum of PC 1   (Torrence & Compo 1998)",
        xlabel = "year", ylabel = "period (years)",
        yscale = log2,
        yticks = ([0.5, 1, 2, 3, 5, 7, 10, 16],
                 ["0.5", "1", "2", "3", "5", "7", "10", "16"]))
    hm = heatmap!(ax2, years, periods_pc, P_norm;
        colormap = :viridis, colorrange = (0, quantile(vec(P_norm), 0.98)))
    Colorbar(fig[2, 2], hm; label = "|W(t,s)|² / σ²")
    hlines!(ax2, [3, 7]; color = :white, linestyle = :dash, linewidth = 1.0)
    text!(ax2, years[1] + 1, 4.5; text = "ENSO band (3–7 yr)",
          fontsize = 11, color = :white)
    band!(ax2, years, coi_pc, fill(maximum(periods_pc), length(years));
          color = (:black, 0.25))
    rowsize!(fig.layout, 1, Fixed(120))

    save(joinpath(FIGDIR, "fig09_pc1_spectrogram.png"), fig)
end

# 10.  Linear Inverse Model (LIM), third method (linear, dynamical)
println("\nFig 10: LIM eigenvalue spectrum + ENSO mode time series")
include("lim.jl")

let
    # Standard LIM setup: top-10 PCs, lag τ = 1 month
    n_pcs = 10
    τ_lag = 1
    lim   = fit_lim(PC[:, 1:n_pcs]; τ = float(τ_lag))

    # Identify ENSO mode (period 2–8 yr, least damped, Im σ > 0)
    period_yrs = [imag(s) == 0 ? Inf : 2π / abs(imag(s)) / 12.0 for s in lim.σ]
    decay_yrs  = [real(s) == 0 ? Inf : -1.0 / real(s) / 12.0      for s in lim.σ]
    cand = findall(i -> 2.0 <= period_yrs[i] <= 8.0 && imag(lim.σ[i]) > 0,
                    eachindex(lim.σ))
    idx = cand[argmin([abs(real(lim.σ[c])) for c in cand])]
    @printf "  ENSO mode: period = %.2f yr, e-fold = %.2f yr\n" period_yrs[idx] decay_yrs[idx]

    # Project state onto ENSO mode → complex time series
    z = project_to_mode(lim, PC[:, 1:n_pcs], idx)
    z_real = real.(z)
    z_imag = imag.(z)

    # Sign-align Re(z) with Niño 3.4 (eigenvector phase is arbitrary)
    if cor(z_real[valid_n], nino34[valid_n]) < 0
        z_real .= -z_real
    end
    ρ_LIM = cor(z_real[valid_n], nino34[valid_n])
    @printf "  cor(Re z_ENSO, Niño 3.4) = %.3f\n" ρ_LIM

    fig = Figure(size = (1300, 540))

    # Left: LIM eigenvalues in the complex plane
    ax1 = Axis(fig[1, 1];
        title  = "LIM eigenvalues σₖ in complex plane",
        xlabel = "Re σ  (1/month) , damping rate",
        ylabel = "Im σ  (1/month) , angular frequency",
        aspect = 1.0)
    re_σ = real.(lim.σ)
    im_σ = imag.(lim.σ)
    scatter!(ax1, re_σ, im_σ; color = :gray, markersize = 12)
    # highlight ENSO pair
    scatter!(ax1, [re_σ[idx], re_σ[idx]], [im_σ[idx], -im_σ[idx]];
             color = :crimson, markersize = 18, marker = :star5,
             label = "ENSO mode (period $(round(period_yrs[idx]; digits=1)) yr)")
    vlines!(ax1, [0]; color = :gray, linewidth = 0.7)
    hlines!(ax1, [0]; color = :gray, linewidth = 0.7)
    # ENSO 3-7 yr band (in 1/month units): Im σ ∈ [2π/(7·12), 2π/(3·12)]
    ω_lo = 2π / (7 * 12); ω_hi = 2π / (3 * 12)
    band!(ax1, [-0.4, 0.05], [ω_lo, ω_lo], [ω_hi, ω_hi];
          color = (:crimson, 0.08))
    band!(ax1, [-0.4, 0.05], [-ω_hi, -ω_hi], [-ω_lo, -ω_lo];
          color = (:crimson, 0.08))
    text!(ax1, -0.35, 0.18; text = "ENSO band\n(3–7 yr)",
          fontsize = 10, color = (:crimson, 0.7))
    axislegend(ax1; position = :rb, framevisible = false)

    # Right: LIM mode time series vs Niño 3.4
    years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]
    ax2 = Axis(fig[1, 2];
        title  = "LIM ENSO mode Re(z) vs Niño 3.4   (ρ = $(round(ρ_LIM; digits=3)))",
        xlabel = "year", ylabel = "anomaly")
    s = std(nino34[valid_n]) / std(z_real)
    lines!(ax2, years, nino34;       color = (:steelblue, 0.85), label = "Niño 3.4")
    lines!(ax2, years, z_real .* s;  color = (:darkorange, 0.85), label = "Re(z) LIM ENSO mode (rescaled)")
    hlines!(ax2, [0]; color = :gray, linewidth = 0.5)
    axislegend(ax2; position = :rt, framevisible = false)

    # Bottom: a 2-D phase portrait of the LIM mode itself,
    # showing the (a, b) trajectory in the eigenvector plane.
    ax3 = Axis(fig[2, 1:2];
        title  = "LIM mode 2-D phase portrait (in-phase vs quadrature), climate state circulating in mode space",
        xlabel = "Re(z)  ←  ENSO amplitude", ylabel = "Im(z)  ←  quadrature")
    sct = scatter!(ax3, z_real, z_imag; color = nino34, colormap = :RdBu_11,
                   colorrange = (-2.5, 2.5), markersize = 4, alpha = 0.7)
    # Mark the strongest historical El Niños
    for d in [Date(1972,12,1), Date(1982,12,1), Date(1997,12,1),
              Date(2015,12,1), Date(2023,12,1)]
        i = findfirst(==(d), Date.(time_w))
        if !isnothing(i)
            scatter!(ax3, [z_real[i]], [z_imag[i]];
                color = :black, markersize = 14, marker = :star5)
            text!(ax3, z_real[i] + 0.5, z_imag[i];
                text = " " * Dates.format(d, "u yyyy"),
                fontsize = 9, color = :black)
        end
    end
    rowsize!(fig.layout, 1, Relative(0.55))
    save(joinpath(FIGDIR, "fig10_lim.png"), fig)

    # Save key LIM numbers
    open(joinpath(FIGDIR, "lim_summary.txt"), "w") do io
        @printf io "## LIM (Linear Inverse Model)\n"
        @printf io "n_PCs = %d, lag τ = %d month(s)\n\n" n_pcs τ_lag
        @printf io "ENSO eigenpair:\n"
        @printf io "  σ = %+.4f %s %.4f i  (1/month)\n" real(lim.σ[idx]) (imag(lim.σ[idx]) >= 0 ? "+" : "-") abs(imag(lim.σ[idx]))
        @printf io "  period            = %.2f years\n" period_yrs[idx]
        @printf io "  e-folding decay   = %.2f years\n" decay_yrs[idx]
        @printf io "  cor(Re z, Niño 3.4) = %.3f\n" ρ_LIM
    end
end

println("\nAll figures saved to $FIGDIR")
