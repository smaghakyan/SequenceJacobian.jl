# Bijection: bidirectional mapping with identity as default.

using OrderedCollections: OrderedSet

struct Bijection{T, S}
    map::Dict{T, S}
    invmap::Dict{S, T}
end

# The one true constructor — always call this with concrete types.
function make_bijection(map::Dict{T, S}) where {T, S}
    clean   = Dict{T, S}(k => v for (k, v) in map if k != v)
    invmap  = Dict{S, T}()
    for (k, v) in clean
        if haskey(invmap, v)
            error("Duplicate value $v for keys $(invmap[v]) and $k")
        end
        invmap[v] = k
    end
    return Bijection{T, S}(clean, invmap)
end

# Public constructors — all routes lead to make_bijection with concrete types.
function Bijection(map::Dict)
    # Dict{Any,Any} or other untyped — concretise to String keys/values
    K2 = (keytype(map) === Any) ? String : keytype(map)
    V2 = (valtype(map) === Any) ? String : valtype(map)
    return make_bijection(Dict{K2, V2}(map))
end

Bijection() = make_bijection(Dict{String,String}())

Base.keytype(b::Bijection{T,S}) where {T,S} = T
Base.valtype(b::Bijection{T,S}) where {T,S} = S
Base.length(b::Bijection) = length(b.map)
Base.iterate(b::Bijection) = iterate(b.map)
Base.iterate(b::Bijection, i) = iterate(b.map, i)
Base.isempty(b::Bijection) = isempty(b.map)

function inv(b::Bijection{T,S}) where {T,S}
    return make_bijection(b.invmap)
end

Base.show(io::IO, b::Bijection{T,S}) where {T,S} = print(io, "Bijection{$T,$S}($(b.map))")

function Base.getindex(b::Bijection, k)
    return get(b.map, k, k)
end

# Compose two bijections: (x * y)(k) = x(y(k))
function Base.:*(x::Bijection{T,S}, y::Bijection{R,T}) where {R,T,S}
    z = Dict{R,S}()
    for (v, u) in x.map
        w = get(y.invmap, v, v)
        z[w] = u
    end
    for (w, v) in y.map
        if v ∉ keys(x.map)
            z[w] = v
        end
    end
    return make_bijection(z)
end

# Apply bijection to dict keys (forward)
function Base.:*(x::Bijection{T,T}, y::Dict{T,S}) where {T,S}
    d = Dict{T,S}()
    for (k, v) in y
        if k in keys(x.map)
            d[x.map[k]] = v
        elseif k ∉ keys(d)
            d[k] = v
        end
    end
    return d
end

# Apply bijection to dict keys (reverse)
function Base.:*(x::Dict{T,S}, y::Bijection{T,T}) where {T,S}
    d = Dict{T,S}()
    for (k, v) in x
        if k in keys(y.map)
            d[y.map[k]] = v
        elseif k ∉ keys(d)
            d[k] = v
        end
    end
    return d
end

Base.:*(x::Bijection{T,S}, y::Vector{T}) where {T,S} = [x[k] for k in y]
Base.:*(x::Vector{T}, y::Bijection{T,S}) where {T,S} = [y[k] for k in x]
Base.:*(x::Bijection{T,S}, y::Set{T}) where {T,S} = Set(x[k] for k in y)
Base.:*(x::Bijection{T,S}, y::OrderedSet{T}) where {T,S} = OrderedSet{S}(x[k] for k in y)
# Apply bijection to ordered set — coerce element type to String
Base.:*(x::OrderedSet{T}, y::Bijection{T,S}) where {T,S} = OrderedSet{S}(y[k] for k in x)
# Generic fallback: coerce each element to String before lookup
function Base.:*(x::OrderedSet, y::Bijection{T,S}) where {T,S}
    OrderedSet{S}(y[string(k)] for k in x)
end

# Generic fallback: Bijection * OrderedSet{Any}
function Base.:*(x::Bijection{T,S}, y::OrderedSet) where {T,S}
    OrderedSet{S}(x[string(k)] for k in y)
end
Base.:*(x::Bijection{T,S}, y::T) where {T,S} = x[y]
Base.:*(x::T, y::Bijection{T,S}) where {T,S} = y[x]
Base.:*(x::Bijection, y::Nothing) = nothing
Base.:*(x::Nothing, y::Bijection) = nothing
