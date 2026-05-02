using LinearAlgebra
using Statistics
using NearestNeighbors

# ============================================================
# Embedding-quality metrics for low-dimensional projections.
# References:
#   J. Venna & S. Kaski, "Neighborhood preservation in nonlinear projection
#   methods: An experimental study", Proc. ICANN 2001.
# ============================================================

"""
    knn_indices(X, k)

For each row of X (N×D matrix of points), return a length-N vector of
length-k integer arrays giving the indices of that row's k nearest
neighbours (excluding itself), in ascending-distance order.
"""
function knn_indices(X::AbstractMatrix{T}, k::Int) where T<:Real
    N = size(X, 1)
    # KDTree expects D×N column-major points
    tree = KDTree(Matrix(X'))
    out = Vector{Vector{Int}}(undef, N)
    @inbounds for i in 1:N
        idxs, _ = knn(tree, X[i, :], k+1, true)   # +1 to exclude self
        # remove self
        out[i] = filter(j -> j != i, idxs)
        # ensure exactly k
        out[i] = out[i][1:min(k, length(out[i]))]
    end
    return out
end

"""
    knn_ranks(X)

For each row i, return a length-N integer vector r where r[j] is the rank
of point j in i's distance ordering (rank 0 = self, rank 1 = nearest neighbour, ...).
Returned as an N×N integer matrix.
"""
function knn_ranks(X::AbstractMatrix{T}) where T<:Real
    N = size(X, 1)
    R = zeros(Int, N, N)
    @inbounds for i in 1:N
        d = vec(sum(abs2, X .- X[i:i, :]; dims=2))
        order = sortperm(d)              # order[1] == i (self)
        for (rank, j) in enumerate(order)
            R[i, j] = rank - 1            # 0 for self, 1 for nearest, ...
        end
    end
    return R
end

"""
    trustworthiness(X_high, X_low, k)

Trustworthiness T(k) ∈ [0, 1].  T = 1 ⇒ no false neighbours appear in the
low-dim embedding's k-NN that weren't already neighbours in high dim;
T < 1 penalises by their high-dim rank.

    T(k) = 1 - (2 / (N k (2N - 3k - 1))) Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)

where U_k(i) is the set of points that are in i's low-dim k-NN but not
its high-dim k-NN, and r(i,j) is j's rank in i's high-dim ordering.
"""
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

"""
    continuity(X_high, X_low, k)

Continuity C(k) ∈ [0, 1].  Symmetric counterpart of trustworthiness:
penalises true high-dim neighbours that *fail* to be neighbours in the
low-dim embedding, weighted by their low-dim rank.
"""
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

# ============================================================
# Procrustes alignment (orthogonal Procrustes) — used to compare two
# embeddings of the same points up to rotation + sign flips, since
# eigenvectors are determined only up to those symmetries.
# ============================================================

"""
    procrustes_align(A, B)

Find orthogonal R minimising ‖A R − B‖_F, return (R, error) where
error = ‖A R − B‖_F / ‖B‖_F.
"""
function procrustes_align(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    F = svd(A' * B)
    R = F.U * F.Vt
    err = norm(A * R - B) / norm(B)
    return R, err
end
