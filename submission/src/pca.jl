using LinearAlgebra
using Statistics

struct PCAResult{T<:Real}
    mean::Matrix{T}
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
    scores::Matrix{T}
    explained::Vector{T}
end

function fit_pca(X::AbstractMatrix{T}; rank::Union{Nothing,Int}=nothing) where T<:Real
    μ = mean(X; dims=1)
    Xc = X .- μ
    F = svd(Xc)
    r = isnothing(rank) ? length(F.S) : min(rank, length(F.S))

    U = F.U[:, 1:r]
    S = F.S[1:r]
    V = F.V[:, 1:r]
    scores = U .* S'

    total_var = sum(abs2, F.S)
    explained = (S .^ 2) ./ total_var

    return PCAResult{T}(Matrix(μ), U, S, V, scores, explained)
end

project_pca(pca::PCAResult, X::AbstractMatrix) = (X .- pca.mean) * pca.V

function denoise_pca(X::AbstractMatrix{T}, k::Int) where T<:Real
    μ = mean(X; dims=1)
    Xc = X .- μ
    F = svd(Xc)
    k = min(k, length(F.S))
    Xrec = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.V[:, 1:k]'
    return Xrec .+ μ
end
