# Misc: assorted utility functions.
# Julia port of Python SSJ utilities/misc.py

using LinearAlgebra

function make_tuple(x)
    # If not tuple or list, make into tuple with one element.
    return (x isa Union{Tuple, AbstractArray}) ? x : (x,)
end

# numeric_primitive is defined in blocks/support/simple_displacement.jl.

demean(x) = x .- sum(x) / length(x)

# LU factorization and solution (matching Python's scipy.linalg.lu_factor / lu_solve)
factor(X) = factorize(X)
factored_solve(Z, y) = Z \ y

function unprime(s::String)
    length(s) >= 2 && s[end-1:end] == "_p" ? s[1:end-2] : s
end

uncapitalize(s::String) = lowercasefirst(s)

function list_diff(l1, l2)
    return [k for k in l1 if !(k in l2)]
end

function dict_diff(d1::Dict, d2::Dict)
    return Dict(k => d1[k] for k in keys(d1) if !(k in keys(d2)))
end

function smart_set(data)
    data isa AbstractString ? Set([data]) : Set(data)
end

function smart_zip(keys, values)
    values isa Float64 ? zip(keys, [values]) : zip(keys, values)
end

function smart_zeros(n)
    n > 1 ? zeros(n) : 0.0
end

# Logit choice probability along dim 1 (matches Python axis=0).
function logit(V, scale)
    Vmax  = maximum(V, dims=1)
    Vnorm = V .- Vmax
    Vexp  = exp.(Vnorm ./ scale)
    return Vexp ./ sum(Vexp, dims=1)
end

# Logsum formula along dim 1.
function logsum(V, scale)
    Vmax  = maximum(V, dims=1)
    Vnorm = V .- Vmax
    return Vmax .+ scale .* log.(sum(exp.(Vnorm ./ scale), dims=1))
end

# Logit choice probabilities and logsum along dim 1.
function logit_choice(V, scale)
    const_ = maximum(V, dims=1)
    Vnorm  = V .- const_
    Vexp   = exp.(Vnorm ./ scale)
    Vexpsum = sum(Vexp, dims=1)
    P  = Vexp ./ Vexpsum
    EV = const_ .+ scale .* log.(Vexpsum)
    return P, EV
end

# Find non-concave region of value function derivative Va.
# Returns (ilower, iupper) bounding the non-concave region.
function nonconcave(Va::AbstractVector)
    nA   = length(Va)
    vmin = Inf
    vmax = -Inf
    for ia in 1:nA-1
        if Va[ia+1] > Va[ia]
            vmin = min(vmin, Va[ia])
            vmax = max(vmax, Va[ia+1])
        end
    end
    ilower = (vmax == -Inf) ? nA : begin
        ia = nA
        while ia > 0; Va[ia] > vmax && break; ia -= 1; end
        ia
    end
    iupper = (vmin == Inf) ? 0 : begin
        ia = 1
        while ia <= nA; Va[ia] < vmin && break; ia += 1; end
        ia
    end
    return ilower, iupper
end


