# Interpolate: efficient linear interpolation exploiting monotonicity.
# Source: FRBNY SSJ.jl — with two bug fixes:
# Those portions: Copyright (c) Federal Reserve Bank of New York, BSD 3-clause license
# (full text and disclaimer in NOTICE.md).
# 1. interpolate_coord_internal: `nx = size(xq,1)` was wrong; now `nx = length(x)`
# 2. interpolate_point: `yq` was undefined; now `y1`

using LinearAlgebra

# ===== Core 1D routines =====

function interpolate_y_internal(x, xq, y)
    xi = 1
    x_low  = x[1]
    x_high = x[2]
    yq = similar(xq)
    for xqi_cur in 1:length(xq)
        xq_cur = xq[xqi_cur]
        while xi < length(x) - 1
            (x_high >= xq_cur) && break
            xi += 1
            x_low  = x_high
            x_high = x[xi + 1]
        end
        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]
    end
    return yq
end

# BUG FIX: original had `nx = size(xq, 1)` for both nx and nxq — nx should be length(x)
function interpolate_coord_internal(x, xq)
    nx  = length(x)    # FIX: was `size(xq, 1)` (wrong)
    nxq = length(xq)

    xi = 1
    x_low  = x[1]
    x_high = x[2]
    xqi  = similar(xq, Int)
    xqpi = similar(xq, Float64)

    for xqi_cur in 1:nxq
        xq_cur = xq[xqi_cur]
        while xi < nx - 1
            (x_high >= xq_cur) && break
            xi += 1
            x_low  = x_high
            x_high = x[xi + 1]
        end
        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur]  = xi
    end
    return xqi, xqpi
end

function apply_coord_internal(x_i, x_pi, y)
    yq = similar(x_pi, Float64)
    for iq in 1:length(x_i)
        y_low  = y[Int(x_i[iq])]
        y_high = y[Int(x_i[iq]) + 1]
        yq[iq] = x_pi[iq] * y_low + (1 - x_pi[iq]) * y_high
    end
    return yq
end

# ===== Broadcasting wrappers (from FRBNY) =====

function _broadcast_wrap_3d(x, xq=nothing, y=nothing)
    ndims(x)==1 && (x = reshape(x, (1, 1, size(x)...)))
    ndims(x)==2 && (x = reshape(x, (1, size(x)...)))
    if !isnothing(xq)
        ndims(xq)==1 && (xq = reshape(xq, (1, 1, size(xq)...)))
        ndims(xq)==2 && (xq = reshape(xq, (1, size(xq)...)))
    end
    if !isnothing(y)
        ndims(y)==1 && (y = reshape(y, 1, 1, size(y)...))
        ndims(y)==2 && (y = reshape(y, 1, size(y)...))
    end
    return x, xq, y
end

function interpolate_y(x, xq, y)
    xq_size = size(xq)
    x, xq, y = _broadcast_wrap_3d(x, xq, y)
    d1 = max(size(x,1), size(xq,1), size(y,1))
    d2 = max(size(x,2), size(xq,2), size(y,2))
    size(x,1)!=d1  && (x  = repeat(x, d1, 1, 1))
    size(xq,1)!=d1 && (xq = repeat(xq, d1, 1, 1))
    size(y,1)!=d1  && (y  = repeat(y, d1, 1, 1))
    size(x,2)!=d2  && (x  = repeat(x, 1, d2, 1))
    size(xq,2)!=d2 && (xq = repeat(xq, 1, d2, 1))
    size(y,2)!=d2  && (y  = repeat(y, 1, d2, 1))
    yq = zeros(d1, d2, xq_size[end])
    for i1 in 1:d1, i2 in 1:d2
        yq[i1,i2,:] = interpolate_y_internal(x[i1,i2,:], xq[i1,i2,:], y[i1,i2,:])
    end
    (d1==1 && d2==1) && return yq[1,1,:]
    d1==1 && return yq[1,:,:]
    d2==1 && return yq[:,1,:]
    return yq
end

function interpolate_coord(x, xq)
    xq_size = size(xq)
    x, xq, _ = _broadcast_wrap_3d(x, xq)
    d1 = max(size(x,1), size(xq,1))
    d2 = max(size(x,2), size(xq,2))
    size(x,1)!=d1  && (x  = repeat(x, d1, 1, 1))
    size(xq,1)!=d1 && (xq = repeat(xq, d1, 1, 1))
    size(x,2)!=d2  && (x  = repeat(x, 1, d2, 1))
    size(xq,2)!=d2 && (xq = repeat(xq, 1, d2, 1))
    xi_out  = zeros(Int,     d1, d2, xq_size[end])
    xpi_out = zeros(Float64, d1, d2, xq_size[end])
    for i1 in 1:d1, i2 in 1:d2
        xi_out[i1,i2,:], xpi_out[i1,i2,:] = interpolate_coord_internal(x[i1,i2,:], xq[i1,i2,:])
    end
    (d1==1 && d2==1) && return xi_out[1,1,:], xpi_out[1,1,:]
    d1==1 && return xi_out[1,:,:], xpi_out[1,:,:]
    d2==1 && return xi_out[:,1,:], xpi_out[:,1,:]
    return xi_out, xpi_out
end

function apply_coord(x_i, x_pi, y)
    xi_size = size(x_i)
    x_i, x_pi, y = _broadcast_wrap_3d(x_i, x_pi, y)
    d1 = max(size(x_i,1), size(x_pi,1), size(y,1))
    d2 = max(size(x_i,2), size(x_pi,2), size(y,2))
    size(x_i,1)!=d1  && (x_i  = repeat(x_i, d1, 1, 1))
    size(x_pi,1)!=d1 && (x_pi = repeat(x_pi, d1, 1, 1))
    size(y,1)!=d1    && (y    = repeat(y, d1, 1, 1))
    size(x_i,2)!=d2  && (x_i  = repeat(x_i, 1, d2, 1))
    size(x_pi,2)!=d2 && (x_pi = repeat(x_pi, 1, d2, 1))
    size(y,2)!=d2    && (y    = repeat(y, 1, d2, 1))
    yq = zeros(Float64, d1, d2, xi_size[end])
    for i1 in 1:d1, i2 in 1:d2
        yq[i1,i2,:] = apply_coord_internal(x_i[i1,i2,:], x_pi[i1,i2,:], y[i1,i2,:])
    end
    (d1==1 && d2==1) && return yq[1,1,:]
    d1==1 && return yq[1,:,:]
    d2==1 && return yq[:,1,:]
    return yq
end

# ===== Robust (binary search) interpolation =====

function interpolate_coord_robust(x, xq; check_increasing=false)
    ndims(x)!=1 && error("x must be 1D in interpolate_coord_robust")
    check_increasing && any(x[1:end-1] .>= x[2:end]) && error("x must be strictly increasing")
    if ndims(xq) == 1
        return interpolate_coord_robust_vector(x, xq)
    else
        i, pi = interpolate_coord_robust_vector(x, vec(xq))
        return reshape(i, size(xq)), reshape(pi, size(xq))
    end
end

function interpolate_coord_robust_vector(x, xq)
    n  = length(x)
    nq = length(xq)
    xqi  = Vector{Int}(undef, nq)
    xqpi = Vector{Float64}(undef, nq)
    for iq in 1:nq
        if xq[iq] < x[1]
            ilow = 1
        elseif xq[iq] > x[end-1]
            ilow = n - 1
        else
            ihigh, ilow = n, 1
            while ihigh - ilow > 1
                imid = (ihigh + ilow) ÷ 2
                xq[iq] > x[imid] ? (ilow = imid) : (ihigh = imid)
            end
        end
        xqi[iq]  = ilow
        xqpi[iq] = (x[ilow+1] - xq[iq]) / (x[ilow+1] - x[ilow])
    end
    return xqi, xqpi
end

# BUG FIX: original had `yq` undefined; changed to `y1`
function interpolate_point(x, x0, x1, y0, y1)
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)  # FIX: was `yq - y0`
end
