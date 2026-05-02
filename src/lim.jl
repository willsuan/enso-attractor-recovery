using LinearAlgebra
using Statistics

# ============================================================
# Linear Inverse Model (LIM)
#
# Reference: Penland & Sardeshmukh (1995), "The optimal growth of
# tropical sea surface temperature anomalies", J. Climate 8, 1999–2024.
#
# Assume the system state x_t evolves as a stationary stochastic linear
# system in continuous time:
#
#       dx/dt = L x + Q ξ(t)
#
# with ξ a vector of unit-variance white noise.  The lag-τ covariance is
#
#       C(τ) = E[x(t+τ) x(t)^T] = exp(L τ) C(0).
#
# Maximum-likelihood estimator of L from data:
#
#       G(τ) = C(τ) C(0)^{-1}     (one-step propagator at lag τ)
#       L    = (1/τ) log G(τ)     (matrix logarithm; principal branch)
#
# Eigendecomposition of L:
#
#       L v_k = σ_k v_k,    σ_k = α_k + i ω_k.
#
# Each eigenpair (σ_k, v_k) is a *normal mode* of the linearised
# climate dynamics:
#
#       e-folding decay time   τ_e = -1 / α_k    (only mode if Im σ_k = 0)
#       oscillation period     T   = 2π / |ω_k|  (if Im σ_k ≠ 0)
#
# For the tropical Pacific the most prominent eigenpair is a complex
# conjugate pair with T ≈ 3–5 yr (the ENSO mode) and α_k ≈ -1/yr
# (e-folding ≈ 1 yr) — i.e., a damped oscillator that is continually
# excited by the Q ξ noise.
# ============================================================

"""
    LIMResult

Result of fitting a linear inverse model.
Fields:
- `L`        : continuous-time generator (size D×D)
- `G`        : discrete-time propagator at lag τ
- `τ`        : lag used (in years)
- `σ`        : eigenvalues of L (complex; D entries)
- `V`        : right eigenvectors of L (columns; D×D matrix, complex)
- `periods`  : 2π / |Im σ|, in years; Inf for purely real eigenvalues
- `decay`    : -1 / Re σ, in years (e-folding times)
- `is_oscil` : Boolean vector, true where Im σ ≠ 0
"""
struct LIMResult
    L::Matrix{ComplexF64}
    G::Matrix{Float64}
    τ::Float64
    σ::Vector{ComplexF64}
    V::Matrix{ComplexF64}
    periods::Vector{Float64}
    decay::Vector{Float64}
    is_oscil::Vector{Bool}
end

"""
    fit_lim(X::AbstractMatrix; τ::Real = 1.0)

Fit a linear inverse model to a (T × D) data matrix X with rows = time
steps and columns = state variables.  τ is the lag in the same time unit
as the rows (e.g. 1.0 if rows are at 1-month intervals and you want τ = 1 month).

Returns a `LIMResult`.  Internally:
1. Centre X.
2. C0 = X' X / (T-1)
3. C1 = X[1:T-τ, :]' X[1+τ:T, :] / (T-τ-1)        (lag-τ covariance)
4. G = C1 / C0                                      (lag-τ propagator)
5. L = (1/τ) · log(G)                              (continuous-time generator)
6. eigendecompose L
"""
function fit_lim(X::AbstractMatrix; τ::Real = 1.0)
    Tn, D = size(X)
    Xc = X .- mean(X; dims=1)

    C0 = (Xc' * Xc) ./ (Tn - 1)

    @assert τ >= 1 "lag must be >= 1 (in row units)"
    τi = round(Int, τ)
    Xa = Xc[1:end-τi, :]
    Xb = Xc[1+τi:end, :]
    C1 = (Xa' * Xb) ./ (size(Xa, 1) - 1)

    G  = C1 / C0
    L  = ComplexF64.(log(G)) ./ τ

    F  = eigen(L)
    σ  = F.values
    V  = F.vectors

    periods  = [iszero(imag(s)) ? Inf : 2π / abs(imag(s)) for s in σ]
    decay    = [iszero(real(s)) ? Inf : -1.0 / real(s)     for s in σ]
    is_oscil = imag.(σ) .!= 0

    return LIMResult(L, G, float(τ), σ, V, periods, decay, is_oscil)
end

"""
    enso_mode(lim::LIMResult; period_window=(2.5, 7.5))

Return the index of the eigenpair whose oscillation period lies in
`period_window` (years) and which has the *least* damping among such pairs
— the ENSO mode under the recharge-oscillator interpretation.
Returns `nothing` if no eigenvalue has a period in the window.
"""
function enso_mode(lim::LIMResult; period_window::Tuple{<:Real,<:Real} = (2.5, 7.5))
    candidates = findall(i -> lim.is_oscil[i] &&
                              period_window[1] <= lim.periods[i] <= period_window[2] &&
                              imag(lim.σ[i]) > 0,           # canonical pair member
                              eachindex(lim.σ))
    isempty(candidates) && return nothing
    # least-damped = largest decay time = smallest |Re σ|
    return candidates[argmin([abs(real(lim.σ[c])) for c in candidates])]
end

"""
    project_to_mode(lim::LIMResult, X::AbstractMatrix, idx::Int)

Project the data X (T × D) onto the LIM eigenmode at index `idx` and
return the complex-valued time series, of length T.  The real part is
the in-phase ENSO index, the imaginary part is the quadrature
component (90° out of phase).
"""
function project_to_mode(lim::LIMResult, X::AbstractMatrix, idx::Int)
    Xc = X .- mean(X; dims=1)
    # left eigenvectors of L: rows of inv(V).
    W = inv(lim.V)
    return Xc * W[idx, :]
end
