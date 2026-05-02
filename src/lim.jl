using LinearAlgebra
using Statistics

# Linear Inverse Model (Penland & Sardeshmukh 1995). Fits the
# stationary linear ODE dx/dt = L x + Q ξ via the propagator
# G(τ) = C(τ) C(0)^{-1} and L = (1/τ) log G(τ).

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

function fit_lim(X::AbstractMatrix; τ::Real = 1.0)
    Tn, D = size(X)
    Xc = X .- mean(X; dims=1)

    C0 = (Xc' * Xc) ./ (Tn - 1)

    @assert τ >= 1 "lag must be >= 1 (in row units)"
    τi = round(Int, τ)
    Xa = Xc[1:end-τi, :]
    Xb = Xc[1+τi:end, :]
    C1 = (Xa' * Xb) ./ (size(Xa, 1) - 1)

    G = C1 / C0
    L = ComplexF64.(log(G)) ./ τ

    F = eigen(L)
    σ = F.values
    V = F.vectors

    periods  = [iszero(imag(s)) ? Inf : 2π / abs(imag(s)) for s in σ]
    decay    = [iszero(real(s)) ? Inf : -1.0 / real(s)    for s in σ]
    is_oscil = imag.(σ) .!= 0

    return LIMResult(L, G, float(τ), σ, V, periods, decay, is_oscil)
end

function enso_mode(lim::LIMResult; period_window::Tuple{<:Real,<:Real} = (2.5, 7.5))
    candidates = findall(i -> lim.is_oscil[i] &&
                              period_window[1] <= lim.periods[i] <= period_window[2] &&
                              imag(lim.σ[i]) > 0,
                              eachindex(lim.σ))
    isempty(candidates) && return nothing
    return candidates[argmin([abs(real(lim.σ[c])) for c in candidates])]
end

function project_to_mode(lim::LIMResult, X::AbstractMatrix, idx::Int)
    Xc = X .- mean(X; dims=1)
    W = inv(lim.V)
    return Xc * W[idx, :]
end
