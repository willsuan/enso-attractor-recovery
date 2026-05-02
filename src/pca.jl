using LinearAlgebra
using Statistics

"""
    PCAResult

A thin container for PCA outputs.
- mean   : 1×B mean spectrum that was subtracted
- U      : N×r left singular vectors (centered scores ÷ σ)
- S      : r singular values
- V      : B×r right singular vectors (loadings / principal components)
- scores : N×r principal-component scores  (= U .* S')
- explained : variance fraction explained by each component (length r)
"""
struct PCAResult{T<:Real}
    mean::Matrix{T}
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
    scores::Matrix{T}
    explained::Vector{T}
end

"""
    fit_pca(X; rank=nothing)

Compute PCA of an N×B data matrix X via thin SVD of the centred data.
If rank is given, truncate to that many components; otherwise keep min(N, B).
Returns a PCAResult.
"""
function fit_pca(X::AbstractMatrix{T}; rank::Union{Nothing,Int}=nothing) where T<:Real
    μ = mean(X; dims=1)
    Xc = X .- μ
    F = svd(Xc)              # Xc = U * Diagonal(S) * V'
    r = isnothing(rank) ? length(F.S) : min(rank, length(F.S))

    U = F.U[:, 1:r]
    S = F.S[1:r]
    V = F.V[:, 1:r]
    scores = U .* S'         # N×r principal-component coordinates

    total_var = sum(abs2, F.S)
    explained = (S .^ 2) ./ total_var

    return PCAResult{T}(Matrix(μ), U, S, V, scores, explained)
end

"""
    project_pca(pca, X)

Project new spectra (N′×B) onto the principal axes of `pca`.
Returns N′×r matrix of scores.
"""
project_pca(pca::PCAResult, X::AbstractMatrix) = (X .- pca.mean) * pca.V

"""
    denoise_pca(X, k)

Reconstruct X from its k-component PCA approximation
(low-rank denoising / Karhunen–Loève truncation).
Returns the denoised N×B matrix.
"""
function denoise_pca(X::AbstractMatrix{T}, k::Int) where T<:Real
    μ = mean(X; dims=1)
    Xc = X .- μ
    F = svd(Xc)
    k = min(k, length(F.S))
    Xrec = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.V[:, 1:k]'
    return Xrec .+ μ
end
