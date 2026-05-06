using FFTW

# Morlet continuous wavelet transform, following Torrence & Compo
# (1998). Uses ω₀ = 6 by default.

function morlet_cwt(x::AbstractVector{<:Real}, dt::Real;
                    ω0::Real = 6.0, dj::Real = 1/12,
                    s0::Real = 2*dt, jtot::Union{Nothing,Int} = nothing)
    N = length(x)
    if jtot === nothing
        jtot = round(Int, log2(N * dt / s0) / dj)
    end

    period_factor = (4π) / (ω0 + sqrt(2 + ω0^2))
    coi_factor    = sqrt(2)

    scales  = s0 .* 2 .^ ((0:jtot) .* dj)
    periods = period_factor .* scales

    x_centered = x .- mean(x)
    Npad = nextpow(2, N)
    xpad = vcat(x_centered, zeros(Npad - N))
    X    = fft(xpad)

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

    W = zeros(ComplexF64, N, length(scales))
    @inbounds for (j, s) in enumerate(scales)
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

    coi = zeros(N)
    for i in 1:N
        d = min(i - 1, N - i)
        coi[i] = period_factor * coi_factor * d * dt
    end

    return W, collect(scales), collect(periods), coi
end

cwt_power(W::AbstractMatrix{<:Complex}) = abs2.(W)
