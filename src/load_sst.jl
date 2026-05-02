using NCDatasets
using Dates
using Statistics
using Printf

const SST_PATH    = joinpath(@__DIR__, "..", "data", "ersst.v5.sst.mnmean.nc")
const NINO34_PATH = joinpath(@__DIR__, "..", "data", "nino34.csv")

"""
    load_ersst()

Load NOAA ERSSTv5 monthly Sea Surface Temperature into memory.
Returns (sst, lat, lon, time):
  sst  :: Array{Float32,3}   shape (T, |lat|, |lon|), missing → NaN
  lat  :: Vector{Float64}    monotonically increasing latitudes (deg N)
  lon  :: Vector{Float64}    longitudes in [0, 360) (deg E)
  time :: Vector{DateTime}
"""
function load_ersst()
    ds = NCDataset(SST_PATH, "r")
    try
        # ERSSTv5 stores SST as (lon, lat, time)
        raw  = ds["sst"][:, :, :]     # NxLat×T or similar; check dims
        lat  = Float64.(ds["lat"][:])
        lon  = Float64.(ds["lon"][:])
        time = ds["time"][:]
        # ERSSTv5: dims are (lon, lat, time) typically;
        # we want (T, lat, lon), so permute.
        # Inspect — assume (lon, lat, time)
        dims = NCDatasets.dimnames(ds["sst"])
        @assert "lat" in dims && "lon" in dims && "time" in dims
        # build a permutation that yields (time, lat, lon)
        perm = (findfirst(==("time"), dims),
                findfirst(==("lat"),  dims),
                findfirst(==("lon"),  dims))
        sst = permutedims(raw, perm)

        # Convert to Float32 array with NaN for missing
        if eltype(sst) <: Union{Missing, AbstractFloat}
            out = Array{Float32}(undef, size(sst))
            @inbounds for i in eachindex(sst)
                v = sst[i]
                out[i] = ismissing(v) ? NaN32 : Float32(v)
            end
            sst = out
        else
            sst = Float32.(sst)
        end

        return sst, lat, lon, time
    finally
        close(ds)
    end
end

"""
    monthly_climatology(sst::AbstractArray{T,3}, time::AbstractVector)

Given a SST cube (time, lat, lon) and the time vector, compute the
monthly climatology = average over all years of January, February, …, December.
Returns clim :: (12, lat, lon).
"""
function monthly_climatology(sst::AbstractArray{T,3}, time::AbstractVector) where T
    months = Dates.month.(time)
    _, La, Lo = size(sst)
    clim = Array{T}(undef, 12, La, Lo)
    @inbounds for m in 1:12
        idx = findall(==(m), months)
        if isempty(idx)
            clim[m, :, :] .= T(NaN)
            continue
        end
        slice = view(sst, idx, :, :)
        clim[m, :, :] = nanmean3(slice)
    end
    return clim
end

# Element-wise mean ignoring NaNs
function nanmean3(A::AbstractArray{T,3}) where T<:AbstractFloat
    _, La, Lo = size(A)
    out = Array{T}(undef, La, Lo)
    @inbounds for j in 1:La, k in 1:Lo
        s = T(0); n = 0
        for i in axes(A, 1)
            v = A[i, j, k]
            if !isnan(v)
                s += v; n += 1
            end
        end
        out[j, k] = n == 0 ? T(NaN) : s / n
    end
    return out
end

"""
    sst_anomalies(sst, time)

Subtract the monthly climatology from sst to give anomalies.
Returns anomaly cube of same shape as sst.
"""
function sst_anomalies(sst::AbstractArray{T,3}, time::AbstractVector) where T
    clim = monthly_climatology(sst, time)
    months = Dates.month.(time)
    anom = similar(sst)
    @inbounds for i in axes(sst, 1)
        m = months[i]
        anom[i, :, :] = sst[i, :, :] .- clim[m, :, :]
    end
    return anom
end

"""
    subset_region(anom, lat, lon; latlim=(-30, 30), lonlim=(120, 280))

Restrict the SST anomaly cube to a lat/lon box.
Default = tropical Pacific 30°S–30°N, 120°E–80°W (= 120°E to 280°E).
Returns (anom_sub, lat_sub, lon_sub).
"""
function subset_region(anom::AbstractArray{T,3}, lat, lon;
                       latlim=(-30, 30), lonlim=(120, 280)) where T
    lat_idx = findall(l -> latlim[1] <= l <= latlim[2], lat)
    # Handle longitude wrapping if needed.  ERSSTv5 lon is 0..360.
    if lonlim[1] <= lonlim[2]
        lon_idx = findall(l -> lonlim[1] <= l <= lonlim[2], lon)
    else
        lon_idx = findall(l -> l >= lonlim[1] || l <= lonlim[2], lon)
    end
    return anom[:, lat_idx, lon_idx], lat[lat_idx], lon[lon_idx]
end

"""
    flatten_to_matrix(anom)

Reshape the (T, lat, lon) anomaly cube into a (T, N) matrix, dropping
any spatial cells whose time series contains NaN (land).  Returns
(X, mask) where X is (T × N_valid) and mask is a (lat, lon) Boolean
matrix marking valid cells.
"""
function flatten_to_matrix(anom::AbstractArray{T,3}) where T
    N1, La, Lo = size(anom)
    flat = reshape(anom, N1, La * Lo)            # T × (Lat·Lon)
    valid = .!any(isnan, flat; dims=1)
    valid_idx = findall(vec(valid))
    X = flat[:, valid_idx]
    mask = reshape(vec(valid), La, Lo)
    return X, mask, valid_idx
end

"""
    cos_lat_weights(lat, lon, mask)

Return the area weight √cos(lat) for every valid cell, in the order
matching the columns of `flatten_to_matrix`.
"""
function cos_lat_weights(lat, lon, mask::AbstractMatrix{Bool})
    La, Lo = size(mask)
    w = Float32[]
    @inbounds for i in 1:La, j in 1:Lo
        if mask[i, j]
            push!(w, sqrt(max(cosd(lat[i]), 0)))
        end
    end
    return w
end

"""
    download_nino34(; force=false)

Fetch NOAA's official Niño 3.4 index file (monthly anomalies).
Source: https://psl.noaa.gov/data/correlation/nina34.anom.data
"""
function download_nino34(; force::Bool=false)
    if !force && isfile(NINO34_PATH) && filesize(NINO34_PATH) > 100
        return NINO34_PATH
    end
    url = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
    download(url, NINO34_PATH)
    return NINO34_PATH
end

"""
    load_nino34()

Parse NOAA's Niño 3.4 index (PSL ASCII format).  Returns
(time::Vector{Date}, value::Vector{Float64}).

The PSL .data format has:
  line 1   : "yyyy yyyy"   (start year, end year)
  lines 2..: "yyyy v_jan v_feb ... v_dec"
  trailing : sentinel line(s)
"""
function load_nino34()
    isfile(NINO34_PATH) || download_nino34()
    raw = readlines(NINO34_PATH)
    # First line: "start_year end_year"
    yr_line = strip(raw[1])
    parts   = split(yr_line)
    yr_start, yr_end = parse(Int, parts[1]), parse(Int, parts[2])
    times = Date[]
    vals  = Float64[]
    sentinel = nothing
    for line in raw[2:end]
        s = strip(line)
        isempty(s) && continue
        toks = split(s)
        # Sentinel value (sometimes -99.99) appears alone
        if length(toks) == 1
            sentinel = parse(Float64, toks[1])
            break
        end
        # Year + 12 monthly values
        length(toks) < 13 && continue
        yr = parse(Int, toks[1])
        if yr < yr_start || yr > yr_end
            continue
        end
        for m in 1:12
            v = parse(Float64, toks[m+1])
            push!(times, Date(yr, m, 1))
            push!(vals,  v)
        end
    end
    # Replace sentinels with NaN if known
    if sentinel !== nothing
        @inbounds for i in eachindex(vals)
            if vals[i] == sentinel
                vals[i] = NaN
            end
        end
    end
    return times, vals
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Loading ERSSTv5 ...")
    sst, lat, lon, time = load_ersst()
    @printf "SST cube:   shape = %s    dtype = %s\n" size(sst) eltype(sst)
    @printf "lat range:  [%.1f, %.1f]   step ≈ %.2f\n" minimum(lat) maximum(lat) (lat[2]-lat[1])
    @printf "lon range:  [%.1f, %.1f]\n" minimum(lon) maximum(lon)
    @printf "time:       %s ... %s   (%d months)\n" time[1] time[end] length(time)

    println("\nComputing anomalies (subtract monthly climatology)...")
    anom = sst_anomalies(sst, time)
    @printf "anomaly mean = %.3g (should be ~0)\n" mean(filter(!isnan, anom))

    println("\nSubsetting tropical Pacific...")
    pac, lat_p, lon_p = subset_region(anom, lat, lon)
    @printf "tropical Pacific shape = %s\n" size(pac)

    println("\nFlattening to matrix...")
    X, mask, idx = flatten_to_matrix(pac)
    @printf "X shape = %s  (T=%d months × N=%d valid sea cells)\n" size(X) size(X,1) size(X,2)
    @printf "land/missing fraction = %.1f%%\n" 100*(1 - size(X,2)/length(mask))

    println("\nDownloading Niño 3.4 index...")
    download_nino34()
    n_t, n_v = load_nino34()
    valid = .!isnan.(n_v)
    @printf "Niño 3.4: %s ... %s   (%d months valid)\n" n_t[findfirst(valid)] n_t[findlast(valid)] count(valid)
end
