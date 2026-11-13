struct IgnoreInt <: Integer
    x::Int
end

# Wraps an 'Int' that behaves like a normal integer in most arithmetic but is treated specially in differentiation or transformation logic.
IgnoreInt(x::IgnoreInt) = IgnoreInt(x.x)


Base.Int(x::IgnoreInt) = Int(x.x)
Base.Float64(x::IgnoreInt) = Float64(x.x)

Base.:-(x::IgnoreInt) = IgnoreInt(-x.x)
Base.:/(x::IgnoreInt, y::IgnoreInt) = IgnoreFloat64(x.x / y.x)
Base.power_by_squaring(x::Integer, p::IgnoreInt) = IgnoreInt(Base.power_by_squaring(x, p.x))
for f ∈ (:+, :-, :*, :^)
    @eval Base.$f(x::IgnoreInt, y::IgnoreInt) = IgnoreInt($f(x.x, y.x))
end
for f ∈ (:<, :>, :(==))
    @eval Base.$f(x::IgnoreInt, y::IgnoreInt) = $f(x.x, y.x) 
end

ss(x::IgnoreInt) = x
(x::IgnoreInt)(y::Any) = x
apply(x::IgnoreInt, f; kwargs...) = ignore(f(x.x; kwargs...))

# Math on IgnoreInt (returns IgnoreFloat64)
Base.log(x::IgnoreInt) = IgnoreFloat64(log(Float64(x.x)))
Base.exp(x::IgnoreInt) = IgnoreFloat64(exp(Float64(x.x)))
Base.sqrt(x::IgnoreInt) = IgnoreFloat64(sqrt(Float64(x.x)))
Base.abs(x::IgnoreInt) = IgnoreInt(abs(x.x))


struct IgnoreFloat64 <: AbstractFloat
    x::Float64
end

# Wraps a 'Float64' similarly to 'IgnoreInt', for ignoring during differentiation.
IgnoreFloat64(x::IgnoreFloat64) = IgnoreFloat64(x.x)
Base.Int(x::IgnoreFloat64) = Int(x.x)
Base.Float64(x::IgnoreFloat64) = x.x

Base.:-(x::IgnoreFloat64) = IgnoreFloat64(-x.x)
for f ∈ (:+, :-, :*, :/, :^)
    @eval Base.$f(x::IgnoreFloat64, y::IgnoreFloat64) = IgnoreFloat64($f(x.x, y.x))
end
for f ∈ (:<, :>, :(==))
    @eval Base.$f(x::IgnoreFloat64, y::IgnoreFloat64) = $f(x.x, y.x) 
end

ss(x::IgnoreFloat64) = x
(x::IgnoreFloat64)(y::Any) = x
apply(x::IgnoreFloat64, f; kwargs...) = ignore(f(x.x; kwargs...))

# Math functions for IgnoreFloat64 — required for log(), exp(), sqrt(), abs() in SimpleBlocks
for f ∈ (:log, :exp, :sqrt, :abs, :sin, :cos, :tan, :sign, :floor, :ceil, :round)
    @eval Base.$f(x::IgnoreFloat64) = IgnoreFloat64($f(x.x))
end
Base.log(b::Number, x::IgnoreFloat64) = IgnoreFloat64(log(b, x.x))
Base.max(x::IgnoreFloat64, y::IgnoreFloat64) = IgnoreFloat64(max(x.x, y.x))
Base.max(x::IgnoreFloat64, y::Number) = IgnoreFloat64(max(x.x, Float64(y)))
Base.max(x::Number, y::IgnoreFloat64) = IgnoreFloat64(max(Float64(x), y.x))
Base.min(x::IgnoreFloat64, y::IgnoreFloat64) = IgnoreFloat64(min(x.x, y.x))
Base.min(x::IgnoreFloat64, y::Number) = IgnoreFloat64(min(x.x, Float64(y)))
Base.min(x::Number, y::IgnoreFloat64) = IgnoreFloat64(min(Float64(x), y.x))
Base.ifelse(cond::Bool, x::IgnoreFloat64, y::IgnoreFloat64) = IgnoreFloat64(ifelse(cond, x.x, y.x))
Base.isnan(x::IgnoreFloat64) = isnan(x.x)
Base.isinf(x::IgnoreFloat64) = isinf(x.x)


Base.promote_rule(::Type{IgnoreInt}, ::Type{Int}) = IgnoreInt 
Base.promote_rule(::Type{IgnoreInt}, ::Type{Float64}) = IgnoreFloat64 
Base.promote_rule(::Type{IgnoreFloat64}, ::Type{Float64}) = IgnoreFloat64
Base.promote_rule(::Type{IgnoreFloat64}, ::Type{Int}) = IgnoreFloat64
Base.promote_rule(::Type{IgnoreFloat64}, ::Type{IgnoreInt}) = IgnoreFloat64


struct IgnoreArray{T, N} <: AbstractArray{T, N}
    x::AbstractArray{T, N}
end

# Array wrapper that carries array-like behavior but can be used to ignore gradients or apply custom transformation logic.
IgnoreVector{T} = IgnoreArray{T, 1}
IgnoreVector(x::AbstractVector) = IgnoreArray(x)
IgnoreMatrix{T} = IgnoreArray{T, 2}
IgnoreMatrix(x::AbstractMatrix) = IgnoreArray(x)

Base.size(x::IgnoreArray) = Base.size(x.x)
Base.size(x::IgnoreArray, d) = Base.size(x.x, d)

Base.length(x::IgnoreArray) = Base.length(x.x)

Base.getindex(x::IgnoreArray, inds...) = Base.getindex(x.x, inds...)
Base.setindex!(x::IgnoreArray, n, inds...) = Base.setindex!(x.x, n, inds...)

Base.BroadcastStyle(::Type{<:IgnoreArray}) = Broadcast.ArrayStyle{IgnoreArray}()

Base.similar(x::IgnoreArray{T, S}) where {T, S} = IgnoreArray{T, S}(similar(x.x))
Base.similar(x::IgnoreArray, dims::NTuple{N, Int}) where N = IgnoreArray(similar(x.x, dims))
function Base.similar(x::IgnoreArray, ::Type{T}, dims::NTuple{N, Int}) where {T, N} 
    return IgnoreArray(similar(x.x, T, dims))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{IgnoreArray}}, ::Type{T}) where T
    IgnoreArray(similar(Array{T}, size(bc)))
end

(x::IgnoreArray)(y::Any) = x
apply(x::IgnoreArray, f; kwargs...) = ignore(f(x.x; kwargs...))

ignore(x::Int) = IgnoreInt(x)
ignore(x::Real) = IgnoreFloat64(x)
ignore(x::AbstractArray) = IgnoreArray(x)
# For Any-typed values (e.g. from SteadyStateDict{String,Any}), unwrap the actual value
function ignore(x::Any)
    prim = numeric_primitive(x)
    if prim isa AbstractArray
        return IgnoreArray(prim)
    elseif prim isa Integer
        return IgnoreInt(Int(prim))
    elseif prim isa Real
        return IgnoreFloat64(Float64(prim))
    else
        error("$(typeof(x)) is not supported by ignore(). Must be a Number or Array.")
    end
end

ss(x::IgnoreArray) = x

Ignore = Union{IgnoreInt, IgnoreFloat64, IgnoreArray}

""" Wraps a time path 'x', along with steady-state 'ss' and initial state 'ss_initial', used for time-displacement modelling. """
struct Displace{T, N} <: AbstractArray{T, N}
    x::AbstractArray{T, N}
    ss
    ss_initial
    name

    Displace{T, N}(x; ss=nothing, ss_initial=nothing, name="UNKNOWN") where {T, N} = new{T, N}(x, ss, ss_initial, name)
end

Displace(x; ss=nothing, ss_initial=nothing, name="UNKNOWN") = Displace{eltype(x), ndims(x)}(x; ss=ss, ss_initial=ss_initial, name=name)

numeric_primitive(x::Union{Int, Float64}) = x
numeric_primitive(x::Union{IgnoreInt, IgnoreFloat64}) = x.x
numeric_primitive(x::Array{<:Number}) = x
numeric_primitive(x::IgnoreArray{<:Number}) = x.x
numeric_primitive(x::NTuple{N, <:Number}) where {N} = x
numeric_primitive(x::Displace) = x.x
# General fallbacks for AbstractFloat, Integer, and Any (e.g. from SteadyStateDict{String,Any})
numeric_primitive(x::AbstractFloat) = Float64(x)
numeric_primitive(x::Integer) = Int(x)
numeric_primitive(x::AbstractArray{<:Number}) = x
numeric_primitive(x::Any) = x  # last resort — let Julia handle type conversion

Base.size(x::Displace) = Base.size(x.x)
Base.size(x::Displace, d) = Base.size(x.x, d)

Base.length(x::Displace) = Base.length(x.x)

Base.getindex(x::Displace, inds...) = Base.getindex(x.x, inds...)
Base.setindex!(x::Displace, n, inds...) = Base.setindex!(x.x, n, inds...)

Base.BroadcastStyle(::Type{<:Displace}) = Broadcast.ArrayStyle{Displace}()

Base.similar(x::Nothing) = nothing
Base.similar(x::Displace{T, N}) where {T, N} = Displace{T, N}(similar(x.x); ss=similar(x.ss), ss_initial=similar(x.ss_initial))
Base.similar(x::Displace, dims::NTuple{N, Int}) where N = Displace(similar(x.x, dims); ss=similar(x.ss), ss_initial=similar(x.ss_initial))
function Base.similar(x::Displace, ::Type{T}, dims::NTuple{N, Int}) where {T, N} 
    return Displace(similar(x.x, T, dims); ss=similar(x.ss), ss_initial=similar(x.ss_initial))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Displace}}, ::Type{T}) where T
    x = find_ss(bc)
    Displace(similar(Array{T}, size(bc)); ss=x.ss, ss_initial=x.ss_initial)
end

find_ss(x) = x
find_ss(r::Displace, rest) = r
find_ss(bc::Base.Broadcast.Broadcasted) = find_ss(bc.args)
find_ss(args::Tuple) = find_ss(find_ss(args[1]), Base.tail(args))

function (x::Displace)(y::Any)
    if y ≠ 0
        if isnothing(x.ss)
            error("Trying to call $(x.name)($y) but steady-state $(x.name) not given!")
        end
        newx = zeros(size(x))
        if y > 0
            newx[1:end-y] = numeric_primitive(x)[y+1:end]
            newx[end-y+1:end] .= x.ss
        else
            newx[-y+1:end] = numeric_primitive(x)[1:end+y]
            newx[1:-y] .= x.ss_initial
        end
        return Displace(newx; ss=x.ss, ss_initial=x.ss_initial)
    else
        return x
    end
end

# Apply a scalar function (log, exp, ...) to a time path, date by date. (The function
# used to be called on the whole path at once, so `apply(x, log)` failed with
# "no method matching log(::Array)" as soon as a block using it was simulated
# nonlinearly.)
function apply(x::Displace, f; kwargs...)
    g(v) = f(v; kwargs...)
    return Displace(g.(numeric_primitive(x));
                    ss = isnothing(x.ss) ? nothing : g(x.ss),
                    ss_initial = isnothing(x.ss_initial) ? nothing : g(x.ss_initial))
end

Base.:+(x::Displace) = x
Base.:-(x::Displace) = Displace(-numeric_primitive(x); ss=-x.ss, ss_initial=-x.ss_initial)
for f ∈ (:+, :-, :*, :/, :^)
    @eval Base.$f(x::Displace, y::Displace) = Displace($f.(numeric_primitive(x), numeric_primitive(y)); ss=$f.(x.ss, y.ss), ss_initial=$f(x.ss_initial, y.ss_initial))
    @eval Base.$f(x::Displace, y::Number) = Displace($f.(numeric_primitive(x), numeric_primitive(y)); ss=$f.(x.ss, numeric_primitive(y)), ss_initial=$f.(x.ss_initial, numeric_primitive(y)))
    @eval Base.$f(x::Number, y::Displace) = Displace($f.(numeric_primitive(x), numeric_primitive(y)); ss=$f.(numeric_primitive(x), y.ss), ss_initial=$f.(numeric_primitive(x), y.ss_initial))
    @eval Base.$f(x::Any, y::Nothing) = nothing
    @eval Base.$f(x::Nothing, y::Any) = nothing
end

""" Represents a generalized directional derivative that accumulates weighted partials across time indices and orders. """
struct AccumulatedDerivative
    elements
    f_value
    _keys
    _fp_values

    function AccumulatedDerivative(; elements=Dict((0, 0) => 1.0), f_value=1.0)
        return new(elements, f_value, collect(keys(elements)), collect(values(elements)))
    end
end

function Base.show(io::IO, x::AccumulatedDerivative) 
    print(io, "AccumulatedDerivative({" * join(["($i, $m): $y" for ((i, m), y) ∈ x.elements], ", ") * "})")
end

# Matches Python's @property ss = ignore(self.f_value)
ss(x::AccumulatedDerivative) = ignore(x.f_value)

function (x::AccumulatedDerivative)(i::Any)
    keys = [(i + j, compute_l(-i, 0, -j, n)) for (j, n) ∈ x._keys]
    return AccumulatedDerivative(elements=Dict(zip(keys, x._fp_values)), f_value=x.f_value)
end

function apply(x::AccumulatedDerivative, f; h=1e-5, kwargs...)
    if f == log
        return AccumulatedDerivative(elements=Dict(zip(x._keys, [1 / x.f_value * y for y ∈ x._fp_values])), f_value=log(x.f_value))
    else
        return AccumulatedDerivative(elements=Dict(zip(x._keys, [(f(x.f_value + h; kwargs...) - f(x.f_value - h; kwargs...)) / (2h) * y for y ∈ x._fp_values])), f_value=f(x.f_value; kwargs...))
    end
end

Base.:+(x::AccumulatedDerivative) = AccumulatedDerivative(elements=Dict(zip(x._keys, +x._fp_values)), f_value=+x.f_value)
Base.:-(x::AccumulatedDerivative) = AccumulatedDerivative(elements=Dict(zip(x._keys, -x._fp_values)), f_value=-x.f_value)

for f ∈ (:+, :-)
    @eval function Base.$f(x::AccumulatedDerivative, y::AccumulatedDerivative)
        elements = copy(x.elements)
        for (i, z) ∈ y.elements
            if i ∈ keys(elements)
                elements[i] = $f(elements[i], z)
                if abs(elements[i]) < 1e-14
                    pop!(elements, i)
                end
            else
                elements[i] = $f(z)
            end
        end
        return AccumulatedDerivative(elements=elements, f_value=$f(x.f_value, y.f_value))
    end
    @eval Base.$f(x::AccumulatedDerivative, y::Number) = AccumulatedDerivative(elements=Dict(zip(x._keys, x._fp_values)), f_value=$f(x.f_value, numeric_primitive(y)))
    @eval Base.$f(x::Number, y::AccumulatedDerivative) = AccumulatedDerivative(elements=Dict(zip(y._keys, $f(y._fp_values))), f_value=$f(numeric_primitive(x), y.f_value))
end

function Base.:*(x::AccumulatedDerivative, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=(x * y.f_value + y * x.f_value).elements, f_value=(x.f_value * y.f_value))
end
function Base.:*(x::AccumulatedDerivative, y::Number)
    return AccumulatedDerivative(elements=Dict(zip(x._keys, x._fp_values * numeric_primitive(y))), f_value=(x.f_value * numeric_primitive(y)))
end
function Base.:*(x::Number, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=Dict(zip(y._keys, numeric_primitive(x) * y._fp_values)), f_value=(numeric_primitive(x) * y.f_value))
end

function Base.:/(x::AccumulatedDerivative, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=((y.f_value * x - x.f_value * y) / (y.f_value^2)).elements, f_value=(x.f_value / y.f_value))
end
function Base.:/(x::AccumulatedDerivative, y::Number)
    return AccumulatedDerivative(elements=Dict(zip(x._keys, x._fp_values / numeric_primitive(y))), f_value=(x.f_value / numeric_primitive(y)))
end
function Base.:/(x::Number, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=Dict(zip(y._keys, -numeric_primitive(x) / y.f_value^2 * y._fp_values)), f_value=(numeric_primitive(x) / y.f_value))
end

function Base.:^(x::AccumulatedDerivative, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=(x.f_value^(y.f_value - 1)*(y.f_value * x + y * x.f_value * log(x.f_value))).elements, f_value=x.f_value^y.f_value)
end
function Base.:^(x::AccumulatedDerivative, y::Number)
    return AccumulatedDerivative(elements=Dict(zip(x._keys, numeric_primitive(y) * x.f_value^numeric_primitive(y-1) * x._fp_values)), f_value=x.f_value^numeric_primitive(y))
end
function Base.:^(x::Number, y::AccumulatedDerivative)
    return AccumulatedDerivative(elements=Dict(zip(y._keys, log(numeric_primitive(x)) * numeric_primitive(x) ^ y.f_value * y._fp_values)), f_value=numeric_primitive(x)^y.f_value)
end

function compute_l(i, m, j, n)
    if i ≥ 0 && j ≥ 0
        return max(m - j, n)
    elseif i ≥ 0 && j ≤ 0
        return max(m, n) + min(i, -j)
    elseif i ≤ 0 && j ≥ 0 && i + j ≥ 0
        return max(m - i - j, n)
    elseif i ≤ 0 && j ≥ 0 && i + j ≤ 0
        return max(n + i + j, m)
    else
        return max(m, n + i)
    end
end

function vectorize_func_over_time(func, args...)
    d_inds = [i for i ∈ collect(1:length(args)) if args[i] isa Displace]
    x_path = []
    for t ∈ collect(1:size(args[d_inds[1]])[1])
        push!(x_path, func([i ∈ d_inds ? args[i][t] : args[i] for i ∈ collect(1:length(args))]...))
    end
    return x_path
end

function apply_function(func, args...; kwargs...)
    if any([x isa Displace for x ∈ args])
        x_path = vectorize_func_over_time(func, args...)
        return Displace(x_path; ss=func([x isa Displace ? x.ss : numeric_primitive(x) for x ∈ args]...)) 
    elseif any([x isa AccumulatedDerivative for x ∈ args])
        error("Have not yet implemented general apply_function functionality for AccumulatedDerivatives")
    else
        return func(args...; kwargs...)
    end
end
