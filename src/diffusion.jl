using LinearAlgebra
using Distances
using Statistics
using Random

# ============================================================
# Anisotropic Diffusion Maps  (Coifman & Lafon 2006)
#
# Given data points x_1, ..., x_N ∈ R^B, build a Gaussian affinity
#
#     k_ε(x_i, x_j) = exp(-‖x_i - x_j‖² / ε)
#
# Form the α-renormalised kernel
#
#     k_ε^{(α)}(x_i, x_j) = k_ε(x_i, x_j) / (q(x_i)^α  q(x_j)^α),
#         q(x) = Σ_j k_ε(x, x_j),
#
# row-normalise to a Markov transition matrix P,
# and take its eigendecomposition.  The diffusion-map embedding is
#
#     Ψ_t(x_i) = (λ_2^t ψ_2(i),  λ_3^t ψ_3(i),  ...,  λ_{d+1}^t ψ_{d+1}(i)).
#
# The choice of α controls which limiting operator is recovered:
#
#     α = 0   →  graph Laplacian on data (density-biased)
#     α = ½  →  Fokker–Planck operator (advection-diffusion in density)
#     α = 1   →  Laplace–Beltrami operator on the data manifold
#                (density of sampling cancels out)
# ============================================================

"""
    DiffusionMap

Container for diffusion-map output.

- ε       : Gaussian kernel bandwidth used
- α       : anisotropy parameter
- λ       : eigenvalues, sorted descending (length d+1; first is the trivial 1)
- Ψ       : N×(d+1) matrix of right eigenvectors of the transition matrix P
- coords  : N×d embedding (λ_k^t ψ_k for k=2..d+1)
- t       : diffusion time used
- q       : N-vector of kernel densities q(x_i)        (cached for Nyström)
- d_α     : N-vector of α-renormalised row sums         (cached for Nyström)
- anchors : N×B matrix of anchor points (cached for Nyström extension)
"""
struct DiffusionMap{T<:Real}
    ε::T
    α::T
    λ::Vector{T}
    Ψ::Matrix{T}
    coords::Matrix{T}
    t::Int
    q::Vector{T}
    d_α::Vector{T}
    anchors::Matrix{T}
end

# ------------------------------------------------------------------
# Bandwidth heuristic — median of squared pairwise distances.
# Coifman et al. recommend choosing ε so that log Σ k_ε vs log ε
# has its maximum slope; the median heuristic is a robust shortcut.
# ------------------------------------------------------------------
"""
    median_bandwidth(D2; q=0.5)

Given a pairwise squared-distance matrix D2 (N×N, zeros on diagonal),
return the q-quantile of the off-diagonal entries.
Default q=0.5 → median.
"""
function median_bandwidth(D2::AbstractMatrix; q::Real=0.5)
    N = size(D2, 1)
    vals = Float64[]
    @inbounds for i in 1:N, j in (i+1):N
        push!(vals, D2[i, j])
    end
    return quantile(vals, q)
end

"""
    pairwise_sqdist(X)

Squared Euclidean distance matrix of the N rows of X (N×B).
Returns N×N symmetric matrix with zero diagonal.
"""
function pairwise_sqdist(X::AbstractMatrix{T}) where T<:Real
    # ‖xi − xj‖² = ‖xi‖² + ‖xj‖² − 2 xi·xj
    G = X * X'
    n2 = diag(G)
    D2 = n2 .+ n2' .- 2 .* G
    @inbounds for i in axes(D2, 1)
        D2[i, i] = zero(T)
    end
    D2 .= max.(D2, zero(T))           # numerical floor
    return D2
end

"""
    fit_diffusion_map(X; α=1.0, ε=nothing, d=10, t=1)

Compute the diffusion-map embedding of data matrix X (N×B).
- α : anisotropy parameter (0, ½, or 1)
- ε : kernel bandwidth (squared distance scale).  If nothing, use median heuristic.
- d : embedding dimension (number of nontrivial eigenvectors kept)
- t : diffusion time (powers eigenvalues to λ^t)

Returns a DiffusionMap.
"""
function fit_diffusion_map(X::AbstractMatrix{T};
                            α::Real=1.0,
                            ε::Union{Nothing,Real}=nothing,
                            d::Int=10,
                            t::Int=1) where T<:Real
    N = size(X, 1)
    D2 = pairwise_sqdist(X)
    ε_used = isnothing(ε) ? median_bandwidth(D2) : T(ε)

    # 1.  Gaussian affinity
    K = exp.(.-(D2 ./ ε_used))

    # 2.  α-renormalisation
    q = vec(sum(K; dims=2))                     # density estimate
    qα = q .^ T(α)
    Kα = K ./ (qα * qα')                        # broadcasting outer product

    # 3.  row-normalise to Markov transition matrix P
    d_α = vec(sum(Kα; dims=2))
    P = Kα ./ d_α                               # row-stochastic

    # 4.  Eigendecomposition.
    #     P is similar to the symmetric matrix M = D^{-1/2} Kα D^{-1/2},
    #     where D = diag(d_α).  We diagonalise M (real symmetric) and
    #     undo the similarity to recover right eigenvectors of P.
    Dα_inv_sqrt = 1 ./ sqrt.(d_α)
    M = (Dα_inv_sqrt .* Kα) .* Dα_inv_sqrt'     # symmetric
    M = (M + M') ./ 2                            # enforce symmetry numerically
    F = eigen(Symmetric(M))

    # eigen returns ascending order — flip to descending
    order = sortperm(F.values; rev=true)
    λ_all = T.(F.values[order])
    V_all = F.vectors[:, order]

    # right eigenvectors of P  =  D^{-1/2} · (eigvecs of M)
    Ψ_all = Dα_inv_sqrt .* V_all

    # keep d+1 leading eigenvectors (first is the trivial λ₁ = 1)
    keep = min(d + 1, length(λ_all))
    λ = λ_all[1:keep]
    Ψ = Ψ_all[:, 1:keep]

    # diffusion-time embedding (skip trivial first component)
    coords = Ψ[:, 2:keep] .* (λ[2:keep] .^ t)'

    return DiffusionMap{T}(T(ε_used), T(α), λ, Ψ, coords, t, q, d_α, Matrix(X))
end

# ============================================================
# Nyström out-of-sample extension.
#
# Given a diffusion map fit on M anchor points and N′ new points y_j,
# extend the embedding to the new points without rebuilding the kernel
# matrix.  The Nyström formula for right eigenvectors of P is
#
#     ψ_k(y) = (1 / λ_k) · Σ_{i=1}^{M}  P̃(y, x_i)  ψ_k(x_i)
#
# where P̃(y, x_i) is the row of the α-normalised transition matrix
# defined the same way as during fitting, but with rows over y.
# ============================================================

"""
    transition_matrix(dm)

Reconstruct the row-stochastic transition matrix P used during fitting,
restricted to the anchors.  Useful for diffusion-time smoothing.
P = D_α^{-1} K_α  where K_α is the α-renormalised affinity matrix.
"""
function transition_matrix(dm::DiffusionMap{T}) where T<:Real
    X = dm.anchors
    D2 = pairwise_sqdist(X)
    K  = exp.(.-(D2 ./ dm.ε))
    qα = dm.q .^ dm.α
    Kα = K ./ (qα * qα')
    d_α = vec(sum(Kα; dims=2))
    return Kα ./ d_α
end

"""
    diffusion_smooth(dm, F; t=2)

Smooth a signal F (N×P matrix; N must equal number of anchors) by
applying t steps of the random walk, F̂ = P^t F.  This is the natural
diffusion-map analogue of low-rank denoising.

Each *column* of F is treated as a function on the anchor graph;
diffusion smoothing averages a function's values over its t-step random
walk neighbourhood.  Eigenfunctions of P with small eigenvalues — i.e.
high-frequency modes on the data graph — are damped by λ_k^t.
"""
function diffusion_smooth(dm::DiffusionMap{T}, F::AbstractMatrix; t::Int=2) where T<:Real
    P = transition_matrix(dm)
    Ft = Matrix(F)
    for _ in 1:t
        Ft = P * Ft
    end
    return Ft
end

"""
    nystrom_extend(dm, Y)

Extend a diffusion map fit on dm.anchors to new points Y (N′×B).
Returns the N′×d embedding of Y in the same coordinates as dm.coords.
"""
function nystrom_extend(dm::DiffusionMap{T}, Y::AbstractMatrix{S}) where {T<:Real, S<:Real}
    X = dm.anchors
    N′, B = size(Y)
    M = size(X, 1)

    # cross-affinity K(y_j, x_i)
    # ‖y - x‖² = ‖y‖² + ‖x‖² − 2 y·x
    YX = Y * X'
    yn = sum(abs2, Y; dims=2)
    xn = sum(abs2, X; dims=2)'
    D2 = yn .+ xn .- 2 .* YX
    D2 .= max.(D2, zero(eltype(D2)))
    K  = exp.(.-(D2 ./ dm.ε))                          # N′×M

    # α-renormalise using the *anchor* densities for x_i
    # and a freshly computed density q(y_j) for the new points.
    qY = vec(sum(K; dims=2))                           # N′ — note: uses *anchor* affinities
    qY_α = qY .^ dm.α
    qX_α = (dm.q) .^ dm.α
    Kα = K ./ (qY_α * qX_α')

    # row-normalise using row-sums of *Kα* over anchors
    d_y = vec(sum(Kα; dims=2))
    P̃   = Kα ./ d_y                                    # N′×M

    # Nyström: ψ_k(y) = (1/λ_k) · P̃ · ψ_k(X)
    keep = length(dm.λ)
    Ψy = P̃ * dm.Ψ                                      # N′×(d+1)
    Ψy = Ψy ./ dm.λ'                                   # divide each col by its eigenvalue

    coords_y = Ψy[:, 2:keep] .* (dm.λ[2:keep] .^ dm.t)'
    return coords_y
end
