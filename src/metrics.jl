using LinearAlgebra
using Statistics
using NearestNeighbors

# Embedding-quality metrics: Venna & Kaski 2001 trustworthiness and
# continuity, plus orthogonal Procrustes alignment.

function knn_indices(X::AbstractMatrix{T}, k::Int) where T<:Real
    N = size(X, 1)
    tree = KDTree(Matrix(X'))
    out = Vector{Vector{Int}}(undef, N)
    @inbounds for i in 1:N
        idxs, _ = knn(tree, X[i, :], k+1, true)
        out[i] = filter(j -> j != i, idxs)
        out[i] = out[i][1:min(k, length(out[i]))]
    end
    return out
end

function knn_ranks(X::AbstractMatrix{T}) where T<:Real
    N = size(X, 1)
    R = zeros(Int, N, N)
    @inbounds for i in 1:N
        d = vec(sum(abs2, X .- X[i:i, :]; dims=2))
        order = sortperm(d)
        for (rank, j) in enumerate(order)
            R[i, j] = rank - 1
        end
    end
    return R
end

function trustworthiness(X_high::AbstractMatrix, X_low::AbstractMatrix, k::Int)
    N = size(X_high, 1)
    @assert size(X_low, 1) == N
    R_high = knn_ranks(X_high)
    nn_high = [findall(r -> 0 < r <= k, R_high[i, :]) for i in 1:N]
    nn_low  = knn_indices(X_low, k)

    s = 0.0
    @inbounds for i in 1:N
        nh = Set(nn_high[i])
        for j in nn_low[i]
            if !(j in nh)
                s += R_high[i, j] - k
            end
        end
    end
    return 1.0 - (2.0 / (N * k * (2N - 3k - 1))) * s
end

function continuity(X_high::AbstractMatrix, X_low::AbstractMatrix, k::Int)
    N = size(X_high, 1)
    @assert size(X_low, 1) == N
    R_low = knn_ranks(X_low)
    nn_high = knn_indices(X_high, k)
    nn_low  = [findall(r -> 0 < r <= k, R_low[i, :]) for i in 1:N]

    s = 0.0
    @inbounds for i in 1:N
        nl = Set(nn_low[i])
        for j in nn_high[i]
            if !(j in nl)
                s += R_low[i, j] - k
            end
        end
    end
    return 1.0 - (2.0 / (N * k * (2N - 3k - 1))) * s
end

function procrustes_align(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    F = svd(A' * B)
    R = F.U * F.Vt
    err = norm(A * R - B) / norm(B)
    return R, err
end
