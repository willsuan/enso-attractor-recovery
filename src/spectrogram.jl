using FFTW

# ============================================================
# Morlet continuous wavelet transform
# Reference: Torrence & Compo (1998), "A Practical Guide to Wavelet Analysis"
# Bull. Amer. Meteor. Soc. 79(1), 61–78.
#
# The Morlet wavelet is
#     ψ₀(η) = π^(-1/4) · exp(i ω₀ η) · exp(-η² / 2)
# with the standard non-dimensional frequency ω₀ = 6.
#
# Continuous wavelet transform of a length-N series x with sampling
# interval δt at scales s_j is computed efficiently via FFT:
#     W(s, t) = IFFT[ X(ω) · Ψ̂*(s ω) ]
# where Ψ̂(sω) is the analytic Morlet kernel in Fourier space.
# ============================================================

"""
    morlet_cwt(x, dt; ω0=6.0, dj=1/12, s0=2*dt, jtot=auto)

Continuous wavelet transform of a real time series `x` with sampling
interval `dt`, using a Morlet wavelet (ω₀ = 6 default).
Returns:
  W      :: Matrix{ComplexF64}   shape (length(x), n_scales) — wavelet coefficients
  scales :: Vector{Float64}      length n_scales
  periods:: Vector{Float64}      ≈ 1.03 · scales (Fourier period for Morlet ω₀=6)
  coi    :: Vector{Float64}      cone-of-influence period at each time index

Arguments
- `dj`   logarithmic spacing of scales (1/12 → 12 voices per octave)
- `s0`   smallest scale (default 2·dt)
- `jtot` number of scales (auto-chosen so largest period ≈ N·dt/2)
"""
function morlet_cwt(x::AbstractVector{<:Real}, dt::Real;
                    ω0::Real = 6.0, dj::Real = 1/12,
                    s0::Real = 2*dt, jtot::Union{Nothing,Int} = nothing)
    N = length(x)
    if jtot === nothing
        jtot = round(Int, log2(N * dt / s0) / dj)
    end

    # Fourier-period factor for Morlet (Torrence & Compo Table 1)
    period_factor = (4π) / (ω0 + sqrt(2 + ω0^2))   # ≈ 1.0330 for ω0=6
    # Cone-of-influence factor
    coi_factor    = sqrt(2)                         # Morlet COI = √2 · s

    scales  = s0 .* 2 .^ ((0:jtot) .* dj)
    periods = period_factor .* scales

    # FFT of zero-mean series, with zero padding to next power of 2
    x_centered = x .- mean(x)
    Npad = nextpow(2, N)
    xpad = vcat(x_centered, zeros(Npad - N))
    X    = fft(xpad)

    # Angular frequencies for FFT bins (Torrence & Compo eq. 5)
    k = collect(0:Npad-1)
    ω = similar(k, Float64)
    half = Npad ÷ 2
    @inbounds for i in eachindex(k)
        if i - 1 <= half
            ω[i] =  2π * (i - 1) / (Npad * dt)
        else
            ω[i] = -2π * (Npad - (i - 1)) / (Npad * dt)
        end
    end

    # CWT via the convolution theorem
    W = zeros(ComplexF64, N, length(scales))
    @inbounds for (j, s) in enumerate(scales)
        # Morlet daughter wavelet Fourier transform (Torrence & Compo eq. 6)
        # Ψ̂(sω) = π^(-1/4) · √(2π s/dt) · exp(-(sω - ω0)² / 2) · H(ω)
        # where H is the Heaviside (analytic wavelet).
        kernel = zeros(ComplexF64, length(ω))
        for i in eachindex(ω)
            if ω[i] > 0
                kernel[i] = π^(-0.25) * sqrt(2π * s / dt) *
                            exp(-0.5 * (s * ω[i] - ω0)^2)
            end
        end
        Wpad = ifft(X .* kernel)
        W[:, j] = Wpad[1:N]
    end

    # Cone-of-influence: the period beyond which edge effects dominate
    coi = zeros(N)
    for i in 1:N
        d = min(i - 1, N - i)
        coi[i] = period_factor * coi_factor * d * dt
    end

    return W, collect(scales), collect(periods), coi
end

"""
    cwt_power(W)

Magnitude squared of CWT coefficients = local wavelet power spectrum.
"""
cwt_power(W::AbstractMatrix{<:Complex}) = abs2.(W)
