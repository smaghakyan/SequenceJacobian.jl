# Optimized routines — Julia port of Python SSJ utilities/optimized_routines.py

# Set x in place to max(x, xmin) along its last dimension, which must be ascending
# (as a savings policy on an asset grid is): the loop stops at the first element above
# xmin. Works for arrays of any dimension (the leading dimensions are flattened).
function setmin(x, xmin)
    x2 = reshape(x, :, size(x, ndims(x)))         # shares memory with x
    _setmin_rows!(x2, xmin)
    return x
end

function _setmin_rows!(x, xmin)
    ni, nj = size(x)
    for i in 1:ni
        for j in 1:nj
            if x[i, j] < xmin
                x[i, j] = xmin
            else
                break   # row is ascending — no more elements below xmin
            end
        end
    end
    return x
end

# Test whether max(|x1 - x2|) <= tol element-wise.
function within_tolerance(x1, x2, tol)
    y1 = vec(x1); y2 = vec(x2)
    for i in 1:length(y1)
        abs(y1[i] - y2[i]) > tol && return false
    end
    return true
end

# Aggregate two paths date by date: Z[t] = sum(X[t, ...] .* Y[t, ...]).
# Time is the first dimension of both arrays (e.g. a distribution path and a
# policy path), matching Python's `fast_aggregate`. The previous version
# contracted over the wrong dimensions and did not return a length-T path.
function fast_aggregate(X, Y)
    T = size(X, 1)
    size(Y, 1) == T || error("fast_aggregate: paths have different lengths ($(T) and $(size(Y, 1)))")
    Xr = reshape(X, T, :)
    Yr = reshape(Y, T, :)
    size(Xr, 2) == size(Yr, 2) || error("fast_aggregate: state spaces differ in size")
    Z = zeros(promote_type(eltype(Xr), eltype(Yr)), T)
    @inbounds for j in 1:size(Xr, 2), t in 1:T
        Z[t] += Xr[t, j] * Yr[t, j]
    end
    return Z
end
