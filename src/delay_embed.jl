"""
Time-delay (Takens) embedding utilities.

Takens (1981): for a smooth flow on a d-dimensional attractor and a
generic observable, the map x_t ↦ (x_t, x_{t-τ}, …, x_{t-kτ}) with
k ≥ 2d embeds the attractor into R^{(k+1) D} (where D is the
observable dimension).
"""

"""
    delay_stack(X, k; τ=1)

Given an N×D data matrix `X` (rows = time, columns = features), return a
(N - k*τ) × ((k+1) D) matrix whose row t is the concatenation
[X_t, X_{t-τ}, X_{t-2τ}, …, X_{t-kτ}].

The first `k*τ` rows are dropped because they don't have full history.
The function also returns the original-row indices that survive
(useful for re-aligning timestamps and validation indices).
"""
function delay_stack(X::AbstractMatrix{T}, k::Int; τ::Int=1) where T<:Real
    N, D = size(X)
    k >= 0 || error("delay k must be >= 0")
    k == 0 && return Matrix(X), collect(1:N)
    new_N = N - k * τ
    new_N > 0 || error("not enough rows for k=$k delay")
    Y = Matrix{T}(undef, new_N, (k+1) * D)
    @inbounds for t in 1:new_N
        for j in 0:k
            src_row = t + k*τ - j*τ
            Y[t, (j*D + 1):((j+1)*D)] = X[src_row, :]
        end
    end
    return Y, collect((k*τ + 1):N)
end
