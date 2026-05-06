# Time-delay (Takens 1981) embedding: x_t ↦ (x_t, x_{t-τ}, ..., x_{t-kτ}).

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
