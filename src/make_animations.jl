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

const FIGDIR = joinpath(@__DIR__, "..", "figures")

# Reload analysis (same setup as enso_figures.jl)
println("Loading SST and computing embeddings...")
sst, lat_all, lon_all, time_all = load_ersst()
keep = findall(t -> Date(1950,1,1) <= Date(t) <= Date(2024,12,31), time_all)
sst_w = sst[keep, :, :]; time_w = time_all[keep]
anom = sst_anomalies(sst_w, time_w)
pac, lat, lon = subset_region(anom, lat_all, lon_all)
X_raw, mask, _ = flatten_to_matrix(pac); weights = cos_lat_weights(lat, lon, mask)
X = X_raw .* reshape(weights, 1, :)

n_t, n_v = load_nino34()
ix = [findfirst(==(Date(t)), n_t) for t in time_w]
nino34 = Float64[isnothing(j) ? NaN : n_v[j] for j in ix]
valid_n = .!isnan.(nino34)

pca = fit_pca(X; rank=10); PC = Float64.(pca.scores)
for k in 1:5
    if cor(PC[valid_n, k], nino34[valid_n]) < 0; PC[:, k] .= -PC[:, k]; end
end

dm = fit_diffusion_map(X; α=1.0, d=15, t=1)
Ψ = Float64.(dm.coords)
for k in 1:5
    if cor(Ψ[valid_n, k], nino34[valid_n]) < 0; Ψ[:, k] .= -Ψ[:, k]; end
end

years = [Dates.year(Date(t)) + (Dates.month(Date(t)) - 1) / 12 for t in time_w]

# Figure 11: Static 3D scatter of the attractor (Ψ₂, Ψ₃, Ψ₄)
println("\nFig 11: 3D attractor scatter")
let
    fig = Figure(size = (1100, 900))
    ax = Axis3(fig[1, 1];
        title  = "3-D climate attractor: (Ψ₂, Ψ₃, Ψ₄), 900 monthly snapshots",
        xlabel = "Ψ₂", ylabel = "Ψ₃", zlabel = "Ψ₄",
        elevation = 0.32, azimuth = 0.6)
    sct = scatter!(ax,
        Ψ[:, 1], Ψ[:, 2], Ψ[:, 3];
        color = nino34, colormap = :RdBu_11,
        colorrange = (-2.5, 2.5), markersize = 8, alpha = 0.75)
    Colorbar(fig[1, 2], sct; label = "Niño 3.4 (°C)", height = Relative(0.6))

    # Mark major historical El Niños with stars
    for d in [Date(1972,12,1), Date(1982,12,1), Date(1997,12,1),
              Date(2015,12,1), Date(2023,12,1)]
        i = findfirst(==(d), Date.(time_w))
        if !isnothing(i)
            scatter!(ax, [Ψ[i, 1]], [Ψ[i, 2]], [Ψ[i, 3]];
                color = :black, markersize = 18, marker = :star5)
        end
    end
    save(joinpath(FIGDIR, "fig11_attractor3d.png"), fig)
end

# Figure 12 (animation): Phase-portrait trajectory tracing,
# 1990-2010, showing two full ENSO cycles travelled by the data.
println("\nFig 12: Animated phase-portrait trajectory (1990-2010)")
let
    cs = findfirst(==(Date(1990, 1, 1)), Date.(time_w))
    ce = findfirst(==(Date(2010, 12, 1)), Date.(time_w))
    seg = cs:ce
    n   = length(seg)

    fig = Figure(size = (1300, 480), backgroundcolor = :white)
    ax_phase = Axis(fig[1, 1];
        title  = "Phase portrait Ψ₂ vs Ψ₃, climate state 1990-2010",
        xlabel = "Ψ₂", ylabel = "Ψ₃")
    xlims!(ax_phase, minimum(Ψ[seg, 1]) - 0.05, maximum(Ψ[seg, 1]) + 0.05)
    ylims!(ax_phase, minimum(Ψ[seg, 2]) - 0.05, maximum(Ψ[seg, 2]) + 0.05)

    ax_ts = Axis(fig[1, 2];
        title  = "Niño 3.4 index (current month highlighted)",
        xlabel = "year", ylabel = "Niño 3.4 (°C)")
    lines!(ax_ts, years[seg], nino34[seg];
        color = (:steelblue, 0.5), linewidth = 1.0)
    hlines!(ax_ts, [0]; color = :gray, linewidth = 0.5)

    # Observable nodes the animation will mutate
    trail_x  = Observable(Float64[])
    trail_y  = Observable(Float64[])
    trail_c  = Observable(Float64[])
    head_pt  = Observable(Point2f[Point2f(Ψ[seg[1], 1], Ψ[seg[1], 2])])
    head_col = Observable([nino34[seg[1]]])
    ts_x     = Observable([years[seg[1]]])
    ts_y     = Observable([nino34[seg[1]]])

    scatter!(ax_phase, trail_x, trail_y;
        color = trail_c, colormap = :RdBu_11, colorrange = (-2.5, 2.5),
        markersize = 4, alpha = 0.55)
    scatter!(ax_phase, head_pt;
        color = head_col, colormap = :RdBu_11, colorrange = (-2.5, 2.5),
        markersize = 18, marker = :circle, strokecolor = :black, strokewidth = 1.5)
    scatter!(ax_ts, ts_x, ts_y;
        color = :crimson, markersize = 12, marker = :circle, strokecolor = :black, strokewidth = 1.0)

    # Render at 12 fps, ~21 seconds for 252 monthly frames
    record(fig, joinpath(FIGDIR, "fig12_trajectory_anim.mp4"), 1:n;
           framerate = 12) do i
        idx = seg[i]
        push!(trail_x[],  Ψ[idx, 1]);     trail_x[] = trail_x[]
        push!(trail_y[],  Ψ[idx, 2]);     trail_y[] = trail_y[]
        push!(trail_c[],  nino34[idx]);   trail_c[] = trail_c[]
        head_pt[]  = [Point2f(Ψ[idx, 1], Ψ[idx, 2])]
        head_col[] = [nino34[idx]]
        ts_x[] = [years[idx]];            ts_y[] = [nino34[idx]]
        ax_phase.title[] = string("Phase portrait Ψ₂ vs Ψ₃, ",
                                   Dates.format(Date(time_w[idx]), "u yyyy"))
    end
end

# Figure 13 (animation): SST anomaly map evolving 1996-2000,
# the build-up, peak, and decay of the 1997-98 super-El Niño.
println("\nFig 13: Animated SST anomaly map (1996-2000)")
let
    cs = findfirst(==(Date(1996, 1, 1)), Date.(time_w))
    ce = findfirst(==(Date(2000, 12, 1)), Date.(time_w))
    seg = cs:ce
    n   = length(seg)

    # Build a per-month grid of anomaly values for plotting
    function anom_grid(t_idx, mask, lat, lon, anom)
        La, Lo = size(mask)
        G = fill(NaN32, La, Lo)
        for i in 1:La, j in 1:Lo
            if mask[i, j]
                G[i, j] = anom[t_idx, i, j]
            end
        end
        return G
    end

    G0 = anom_grid(seg[1], mask, lat, lon, pac)
    vmax = 4.0     # °C, fixed scale for visual comparability across frames

    fig = Figure(size = (1300, 540), backgroundcolor = :white)
    ax = Axis(fig[1, 1];
        title  = "SST anomalies, 1996-2000  (build-up of the 1997-98 super-El-Niño)",
        xlabel = "longitude (°E)", ylabel = "latitude (°N)",
        aspect = 2.5)
    G_obs   = Observable(G0)

    hm = heatmap!(ax, lon, lat, lift(g -> g', G_obs);
        colormap = :RdBu_11, colorrange = (-vmax, vmax))
    Colorbar(fig[1, 2], hm; label = "SST anomaly (°C)")

    # Niño 3.4 box overlay
    poly!(ax, Point2f[(190, -5), (240, -5), (240, 5), (190, 5)];
          color = (:black, 0), strokecolor = :black, strokewidth = 2,
          linestyle = :dash)

    record(fig, joinpath(FIGDIR, "fig13_sst_anim.mp4"), 1:n;
           framerate = 8) do i
        idx = seg[i]
        G_obs[] = anom_grid(idx, mask, lat, lon, pac)
        ax.title[] = string("SST anomalies ",
                             Dates.format(Date(time_w[idx]), "u yyyy"))
    end
end

println("\nNew figures saved.")
