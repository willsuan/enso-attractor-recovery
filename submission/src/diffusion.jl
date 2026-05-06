using LinearAlgebra
using Distances
using Statistics
using Random

# Anisotropic diffusion maps (Coifman & Lafon 2006). Builds a Gaussian
# affinity, applies α-renormalisation k_ε^(α) = k_ε / (q^α q^α), then
# row-normalises to a Markov transition matrix P. The leading
# eigenvectors of P parameterise the manifold; α controls the limiting
# operator (0 → graph Laplacian, ½ → Fokker-Planck, 1 → Laplace-
# Beltrami on the data manifold).

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

function median_bandwidth(D2::AbstractMatrix; q::Real=0.5)
    N = size(D2, 1)
    vals = Float64[]
    @inbounds for i in 1:N, j in (i+1):N
        push!(vals, D2[i, j])
    end
    return quantile(vals, q)
end

function pairwise_sqdist(X::AbstractMatrix{T}) where T<:Real
    G = X * X'
    n2 = diag(G)
    D2 = n2 .+ n2' .- 2 .* G
    @inbounds for i in axes(D2, 1)
        D2[i, i] = zero(T)
    end
    D2 .= max.(D2, zero(T))
    return D2
end

function fit_diffusion_map(X::AbstractMatrix{T};
                            α::Real=1.0,
                            ε::Union{Nothing,Real}=nothing,
                            d::Int=10,
                            t::Int=1) where T<:Real
    N = size(X, 1)
    D2 = pairwise_sqdist(X)
    ε_used = isnothing(ε) ? median_bandwidth(D2) : T(ε)

    K = exp.(.-(D2 ./ ε_used))

    q = vec(sum(K; dims=2))
    qα = q .^ T(α)
    Kα = K ./ (qα * qα')

    d_α = vec(sum(Kα; dims=2))
    P = Kα ./ d_α

    # P is similar to the symmetric M = D^(-1/2) Kα D^(-1/2). Diagonalising
    # M is real-symmetric and stable; we then undo the similarity to get
    # right eigenvectors of P via ψ_k = D^(-1/2) v_k.
    Dα_inv_sqrt = 1 ./ sqrt.(d_α)
    M = (Dα_inv_sqrt .* Kα) .* Dα_inv_sqrt'
    M = (M + M') ./ 2
    F = eigen(Symmetric(M))

    order = sortperm(F.values; rev=true)
    λ_all = T.(F.values[order])
    V_all = F.vectors[:, order]

    Ψ_all = Dα_inv_sqrt .* V_all

    keep = min(d + 1, length(λ_all))
    λ = λ_all[1:keep]
    Ψ = Ψ_all[:, 1:keep]

    coords = Ψ[:, 2:keep] .* (λ[2:keep] .^ t)'

    return DiffusionMap{T}(T(ε_used), T(α), λ, Ψ, coords, t, q, d_α, Matrix(X))
end

function transition_matrix(dm::DiffusionMap{T}) where T<:Real
    X = dm.anchors
    D2 = pairwise_sqdist(X)
    K  = exp.(.-(D2 ./ dm.ε))
    qα = dm.q .^ dm.α
    Kα = K ./ (qα * qα')
    d_α = vec(sum(Kα; dims=2))
    return Kα ./ d_α
end

function diffusion_smooth(dm::DiffusionMap{T}, F::AbstractMatrix; t::Int=2) where T<:Real
    P = transition_matrix(dm)
    Ft = Matrix(F)
    for _ in 1:t
        Ft = P * Ft
    end
    return Ft
end

# Nyström extension: ψ_k(y) = (1/λ_k) Σᵢ P̃(y, xᵢ) ψ_k(xᵢ).
function nystrom_extend(dm::DiffusionMap{T}, Y::AbstractMatrix{S}) where {T<:Real, S<:Real}
    X = dm.anchors
    N′, B = size(Y)
    M = size(X, 1)

    YX = Y * X'
    yn = sum(abs2, Y; dims=2)
    xn = sum(abs2, X; dims=2)'
    D2 = yn .+ xn .- 2 .* YX
    D2 .= max.(D2, zero(eltype(D2)))
    K  = exp.(.-(D2 ./ dm.ε))

    qY = vec(sum(K; dims=2))
    qY_α = qY .^ dm.α
    qX_α = (dm.q) .^ dm.α
    Kα = K ./ (qY_α * qX_α')

    d_y = vec(sum(Kα; dims=2))
    P̃   = Kα ./ d_y

    keep = length(dm.λ)
    Ψy = P̃ * dm.Ψ
    Ψy = Ψy ./ dm.λ'

    coords_y = Ψy[:, 2:keep] .* (dm.λ[2:keep] .^ dm.t)'
    return coords_y
end
