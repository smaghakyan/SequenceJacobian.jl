# ImpulseDict: impulse responses (paths of length T), stored as deviations from steady state.
#
# `toplevel`  : variable name => path (Vector of length T)
# `internals` : block name => Dict(name => array whose FIRST dimension is time)
# `T`         : horizon
#
# Every ImpulseDict is stored as ImpulseDict{String,Any}. Earlier versions kept the
# element type of whatever Dict was passed in, which made subsetting
# (`irf[["Y","C"]]`) and unions fail with method errors whenever two differently
# typed dictionaries met.

mutable struct ImpulseDict{R, S} <: ResultDict{R, S}
    toplevel::Dict{R, S}
    internals
    T

    ImpulseDict{R, S}(toplevel::Dict{R, S}, internals, T) where {R, S} = new{R, S}(toplevel, internals, T)
end

function infer_length(toplevel::AbstractDict)
    isempty(toplevel) && error("Cannot infer the horizon of an empty ImpulseDict: pass T explicitly.")
    lengths = [length(v) for v ∈ values(toplevel)]
    maxlength = maximum(lengths)
    minlength = minimum(lengths)
    if maxlength != minlength
        error("Building ImpulseDict with inconsistent lengths $(maxlength) and $(minlength)")
    end
    return maxlength
end
infer_length(i::ImpulseDict) = infer_length(i.toplevel)

# Main constructor: accepts any dictionary (or another ImpulseDict, which is copied).
function ImpulseDict(data::AbstractDict; internals = nothing, T = nothing)
    if data isa ImpulseDict
        isnothing(internals) || error("Supplying ImpulseDict and also internals to constructor not allowed")
        (isnothing(T) || T == data.T) || error("Supplying ImpulseDict and a different T to constructor not allowed")
        return ImpulseDict{String, Any}(Dict{String, Any}(string(k) => v for (k, v) ∈ data.toplevel),
                                        copy(data.internals), data.T)
    end
    if data isa ResultDict
        isnothing(internals) || error("Supplying $(nameof(typeof(data))) and also internals to constructor not allowed")
        top = Dict{String, Any}(string(k) => v for (k, v) ∈ data.toplevel)
        int = copy(data.internals)
    else
        top = Dict{String, Any}(string(k) => v for (k, v) ∈ pairs(data))
        int = isnothing(internals) ? Dict{String, Any}() : internals
    end
    _T = isnothing(T) ? infer_length(top) : T
    return ImpulseDict{String, Any}(top, int, _T)
end

# Typed spellings are kept so that older call sites continue to work.
ImpulseDict{R, S}(data::AbstractDict; internals = nothing, T = nothing) where {R, S} =
    ImpulseDict(data; internals = internals, T = T)

Base.copy(i::ImpulseDict) = typeof(i)(copy(i.toplevel), copy(i.internals), i.T)

function Base.show(io::IO, i::ImpulseDict)
    print(io, "<ImpulseDict: $(sort!(collect(keys(i.toplevel))))",
          isempty(i.internals) ? "" : ", internals=$(collect(keys(i.internals)))", ", T=$(i.T)>")
end
Base.show(io::IO, ::MIME"text/plain", i::ImpulseDict) = show(io, i)

# ----- indexing ----------------------------------------------------------------
# One name returns the path; several names return a smaller ImpulseDict.
Base.getindex(i::ImpulseDict, k::String) = i.toplevel[k]
Base.getindex(i::ImpulseDict, k::AbstractString) = i.toplevel[String(k)]
Base.getindex(i::ImpulseDict, k::Symbol) = i.toplevel[string(k)]
Base.getindex(i::ImpulseDict, k::Tuple) = _subset(i, k)
Base.getindex(i::ImpulseDict, k) = _subset(i, k)

function _subset(i::ImpulseDict, ks)
    isnothing(ks) && return copy(i)
    d = Dict{String, Any}()
    for k ∈ ks
        ks_ = string(k)
        haskey(i.toplevel, ks_) || error("Key \"$(ks_)\" not found in ImpulseDict. Available: $(sort!(collect(keys(i.toplevel))))")
        d[ks_] = i.toplevel[ks_]
    end
    return ImpulseDict{String, Any}(d, Dict{String, Any}(), i.T)
end

# `get` fills missing paths with zeros (used when a target does not respond at all).
Base.get(i::ImpulseDict, k::AbstractString) = get(i.toplevel, String(k), zeros(i.T))
function Base.get(i::ImpulseDict, ks)
    d = Dict{String, Any}(string(k) => get(i.toplevel, string(k), zeros(i.T)) for k ∈ ks)
    return ImpulseDict{String, Any}(d, Dict{String, Any}(), i.T)
end

# ----- arithmetic --------------------------------------------------------------
function unary_operation(i::ImpulseDict, op)
    toplevel = Dict{String, Any}(k => op.(v) for (k, v) ∈ i.toplevel)
    internals = Dict{String, Any}()
    for (b, d) ∈ i.internals
        internals[b] = Dict{String, Any}(k => op.(v) for (k, v) ∈ d)
    end
    return ImpulseDict{String, Any}(toplevel, internals, i.T)
end

# A steady-state array has no time dimension; a path has time as its first one.
_align_ss(path, ssval) = (ssval isa AbstractArray && ndims(path) == ndims(ssval) + 1) ?
                         reshape(ssval, 1, size(ssval)...) : ssval

function binary_operation(a::ImpulseDict, b::Union{ImpulseDict, SteadyStateDict}, op)
    toplevel = Dict{String, Any}(k => op.(v, _align_ss(v, b.toplevel[k])) for (k, v) ∈ a.toplevel)
    internals = Dict{String, Any}()
    for (blk, d) ∈ a.internals
        if haskey(b.internals, blk)
            bd = b.internals[blk]
            internals[blk] = Dict{String, Any}(k => (haskey(bd, k) ? op.(v, _align_ss(v, bd[k])) : v) for (k, v) ∈ d)
        else
            internals[blk] = d
        end
    end
    return ImpulseDict{String, Any}(toplevel, internals, a.T)
end

function binary_operation(a::ImpulseDict, b::Number, op)
    toplevel = Dict{String, Any}(k => op.(v, b) for (k, v) ∈ a.toplevel)
    internals = Dict{String, Any}()
    for (blk, d) ∈ a.internals
        internals[blk] = Dict{String, Any}(k => op.(v, b) for (k, v) ∈ d)
    end
    return ImpulseDict{String, Any}(toplevel, internals, a.T)
end

function binary_operation(a::Number, b::ImpulseDict, op)
    toplevel = Dict{String, Any}(k => op.(a, v) for (k, v) ∈ b.toplevel)
    internals = Dict{String, Any}()
    for (blk, d) ∈ b.internals
        internals[blk] = Dict{String, Any}(k => op.(a, v) for (k, v) ∈ d)
    end
    return ImpulseDict{String, Any}(toplevel, internals, b.T)
end

Base.:+(i::ImpulseDict) = i
Base.:-(i::ImpulseDict) = unary_operation(i, -)
Base.abs(i::ImpulseDict) = unary_operation(i, abs)

for f ∈ (:+, :-, :*, :/)
    @eval Base.$f(i::ImpulseDict, j::ImpulseDict) = binary_operation(i, j, $f)
    @eval Base.$f(i::ImpulseDict, j::SteadyStateDict) = binary_operation(i, j, $f)
    @eval Base.$f(i::ImpulseDict, j::Number) = binary_operation(i, j, $f)
    @eval Base.$f(i::Number, j::ImpulseDict) = binary_operation(i, j, $f)
end

function Base.:*(i::ImpulseDict, x::Bijection)
    newi = copy(i)
    newi.toplevel = x * i.toplevel
    return newi
end
Base.:*(x::Bijection, i::ImpulseDict) = i * x

# ----- stacking ----------------------------------------------------------------
# Stack the paths into one vector. A Dict has no reliable order, so the order is
# always explicit: `force_order` if given, otherwise alphabetical.
function pack(i::ImpulseDict; force_order = nothing)
    T = i.T
    order = isnothing(force_order) ? sort!(collect(keys(i.toplevel))) : [string(k) for k ∈ force_order]
    bigv = zeros(Float64, T * length(order))
    for (n, k) ∈ enumerate(order)
        v = i.toplevel[k]
        bigv[(n-1)*T+1:n*T] .= v
    end
    return bigv
end

function unpack(bigv, outputs, T)
    toplevel = Dict{String, Any}()
    for (n, o) ∈ enumerate(outputs)
        toplevel[string(o)] = bigv[(n-1)*T+1:n*T]
    end
    return ImpulseDict{String, Any}(toplevel, Dict{String, Any}(), T)
end
