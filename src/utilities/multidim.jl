# Multidim: operations on multidimensional arrays for distribution iteration.
# Source: FRBNY SSJ.jl — with fix: replaced `OMEinsum` (extra dependency) with
# Those portions: Copyright (c) Federal Reserve Bank of New York, BSD 3-clause license
# (full text and disclaimer in NOTICE.md).
# manual einsum using LinearAlgebra.

using LinearAlgebra

function make_swap_perm(s1, s2, nD)
    result = collect(1:nD)
    result[s1] = s2
    result[s2] = s1
    return Tuple(result)
end

function multiply_ith_dimension(Pi, i, X)
    perm = make_swap_perm(1, i, ndims(X))
    X = permutedims(X, perm)
    shape = size(X)
    X = reshape(X, (shape[1], :))
    X = Pi * X
    X = reshape(X, (size(Pi, 1), shape[2:end]...))
    return permutedims(X, perm)
end

function outer(pis)
    _pi = pis[1]
    for pi_i in pis[2:end]
        _pi = kron(_pi, pi_i)
    end
    _pi = reshape(_pi, reverse(length.(pis))...)
    return permutedims(_pi, collect(ndims(_pi):-1:1))
end

# Replacement for `ein"ijl,jl->il"(P, X)` without OMEinsum.
# P: (i, j, b), X: (j, b) -> result: (i, b). Written as plain loops: taking array slices inside the
# loop makes the forward iteration of a StageBlock with a discrete choice about ten
# times slower.
function _einsum_ijb_jb_ib(P::AbstractArray{<:Any,3}, X::AbstractMatrix)
    ni, nj, nb = size(P)
    result = zeros(promote_type(eltype(P), eltype(X)), ni, nb)
    @inbounds for b in 1:nb, j in 1:nj
        x = X[j, b]
        for i in 1:ni
            result[i, b] += P[i, j, b] * x
        end
    end
    return result
end

function batch_multiply_ith_dimension(P, i, X)
    P = permutedims(P, make_swap_perm(2, 1+i, ndims(P)))
    X = permutedims(X, make_swap_perm(1, i, ndims(X)))
    Pshape = size(P)
    P = reshape(P, (Pshape[1], Pshape[2], :))
    X = reshape(X, (size(X, 1), :))
    result = _einsum_ijb_jb_ib(P, X)
    result = reshape(result, (Pshape[1], Pshape[3:end]...))
    return permutedims(result, make_swap_perm(1, i, ndims(result)))
end
