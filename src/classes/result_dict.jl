# Abstract type ResultDict that wraps results in top-levels and internals
abstract type ResultDict{R, S} <: AbstractDict{R, S} end

function Base.show(io::IO, r::ResultDict)
    if !isempty(r.internals)
        print(io, "<$(nameof(typeof(r))): $(keys(r.toplevel)), internals=$(keys(r.internals))>")
    else
        print(io, "<$(nameof(typeof(r))): $(keys(r.toplevel)))>")
    end 
end

# Rest of file defines standard operations with ResultDicts: union, merging, copying, etc
Base.keytype(r::ResultDict{R, S}) where {R, S} = R
Base.valtype(r::ResultDict{R, S}) where {R, S} = S
    
# Iteration behavior
Base.iterate(r::ResultDict) = iterate(r.toplevel)
Base.iterate(r::ResultDict, i) = iterate(r.toplevel, i)

Base.getindex(r::ResultDict, k::String; kwargs...) = r.toplevel[k]
Base.getindex(r::ResultDict, k::Tuple; kwargs...) = error("Key $k to $(nameof(typeof(r))) cannot be a tuple")
function Base.getindex(r::T, k; kwargs...) where T <: ResultDict
    # k is an iterable of string keys — extract available keys, error on missing
    try
        d = Dict{String,Any}(string(ki) => r.toplevel[string(ki)] for ki ∈ k)
        return T(d; kwargs...)
    catch e
        if e isa KeyError
            missing_key = e.key
            error("Key \"$(missing_key)\" not found in $(nameof(typeof(r))). Available: $(collect(keys(r.toplevel))[1:min(5,end)])")
        end
        error("Key $k to $(nameof(typeof(r))) needs to be a string or an iterable of strings")
    end
end

Base.setindex!(r::ResultDict, v, k) = r.toplevel[k] = v
Base.haskey(r::ResultDict, k) = haskey(r.toplevel, k)
Base.get(r::ResultDict, k, default) = get(r.toplevel, k, default)

function Base.:*(r::ResultDict, x::Bijection)
    isempty(x) && return r      # identity map: nothing to rename (and no deep copy of the distribution)
    newr = deepcopy(r)
    newr.toplevel = x * r.toplevel
    return newr
end
Base.:*(x::Bijection, r::ResultDict) = r * x

Base.length(r::ResultDict) = length(r.toplevel)

function Base.union(r::ResultDict, s::ResultDict)
    if nameof(typeof(r)) != nameof(typeof(s))
        error("Trying to merge a $(nameof(typeof(r))) with a $(nameof(typeof(s))).")
    end
    merged = copy(r)
    merge!(merged, s)
    return merged
end

Base.keys(r::ResultDict) = keys(r.toplevel)
Base.values(r::ResultDict) = values(r.toplevel)
Base.pairs(r::ResultDict) = pairs(r.toplevel)

function Base.merge!(r::ResultDict, s::ResultDict)
    merge!(r.toplevel, s.toplevel)
    merge!(r.internals, s.internals)
end

# merge! with coercion: accept any dict and convert to matching key/value types
function Base.merge!(r::ResultDict, s::AbstractDict)
    for (k, v) in s
        r.toplevel[string(k)] = v
    end
end
Base.merge!(r::ResultDict, s::Any) = merge!(r.toplevel, Dict(s))

function Base.copy(r::T) where T <: ResultDict
    return T(copy(r.toplevel); internals=copy(r.internals))
end
