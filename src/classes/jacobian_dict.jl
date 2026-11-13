# JacobianDict and FactoredJacobianDict.
# Source: FRBNY SSJ.jl — with bug fixes:
# Those portions: Copyright (c) Federal Reserve Bank of New York, BSD 3-clause license
# (full text and disclaimer in NOTICE.md).
# 1. `set(J.inputs)` -> `Set(J.inputs)` in apply()
# 2. `x.t` -> `x.T` (lowercase typo) in apply()

Jacobian = Union{AbstractArray, IdentityMatrix, SimpleSparse}

Base.copy(::Nothing) = nothing

abstract type NestedDict{R,S} <: AbstractDict{R,S} end

Base.keytype(nd::NestedDict{R,S}) where {R,S} = R
Base.valtype(nd::NestedDict{R,S}) where {R,S} = S
Base.show(io::IO, nd::NestedDict) = print(io, "<$(typeof(nd)) outputs=$(nd.outputs), inputs=$(nd.inputs)>")
# Iterate over output names (matches Python __iter__ behavior)
Base.iterate(nd::NestedDict)    = iterate(nd.outputs)
Base.iterate(nd::NestedDict, i) = iterate(nd.outputs, i)

function Base.union(nd::NestedDict, s)
    merged = JacobianDict(nd.nesteddict; outputs=nd.outputs, inputs=nd.inputs)
    merge!(merged, s)
    return merged
end

Base.getindex(nd::NestedDict, k::AbstractString) = nd.nesteddict[k]

function Base.getindex(nd::T, k::Tuple) where T <: NestedDict
    o, i = k
    o = (o isa Colon) ? nd.outputs : o
    i = (i isa Colon) ? nd.inputs  : i
    if o isa AbstractString
        i isa AbstractString ? nd.nesteddict[o][i] : subdict(nd.nesteddict[o], i)
    else
        i = i isa AbstractString ? (i,) : i
        o_str = OrderedSet{String}(string(oo) for oo in o)
        i_str = i isa Colon ? nd.inputs :
                (i isa AbstractString ? OrderedSet{String}([string(i)]) :
                 OrderedSet{String}(string(ii) for ii in i))
        # Use JacobianDict (coercing constructor) not T to avoid type param issues
        JacobianDict(
            Dict{String,Any}(string(oo) => subdict(nd.nesteddict[string(oo)], i_str)
                             for oo in o_str);
            outputs=o_str, inputs=i_str)
    end
end

function Base.getindex(nd::T, k::Union{OrderedSet,Set,Vector}) where T <: NestedDict
    k_str = OrderedSet{String}(string(kk) for kk in k)
    JacobianDict(
        Dict{String,Any}(string(oo) => nd.nesteddict[string(oo)] for oo in k_str);
        outputs=k_str, inputs=nd.inputs)
end

Base.get(nd::NestedDict, args...; kwargs...) = get(nd.nesteddict, args...; kwargs...)
Base.length(nd::NestedDict)  = length(nd.nesteddict)
Base.keys(nd::NestedDict)    = keys(nd.nesteddict)
Base.values(nd::NestedDict)  = values(nd.nesteddict)
Base.pairs(nd::NestedDict)   = pairs(nd.nesteddict)
Base.copy(nd::NestedDict)    = typeof(nd)(nd)

function Base.merge!(nd::NestedDict, J::NestedDict)
    (isempty(J.outputs) || isempty(J.inputs)) && return nothing
    # Allow merging when inputs differ — just add all inputs
    !isempty(Set(nd.outputs) ∩ Set(J.outputs)) && error("Cannot merge with overlapping outputs")
    nd.outputs = OrderedSet{String}(string(x) for x in nd.outputs ∪ J.outputs)
    nd.inputs  = OrderedSet{String}(string(x) for x in nd.inputs  ∪ J.inputs)
    nd.nesteddict = merge(nd.nesteddict, J.nesteddict)
end

function complete(nd::NestedDict, filler)
    d = Dict{String,Any}()
    for o in nd.outputs
        d[string(o)] = Dict{String,Any}(nd.nesteddict[string(o)])
        for i in nd.inputs
            i ∉ keys(d[o]) && (d[o][i] = filler)
        end
    end
    return typeof(nd)(d; outputs=nd.outputs, inputs=nd.inputs)
end

deduplicate(v::AbstractVector) = unique(v)
subdict(d, ks) = Dict{String,Any}(string(k) => d[string(k)] for k in ks if string(k) in keys(d))

mutable struct JacobianDict{R,S} <: NestedDict{R,S}
    nesteddict; outputs; inputs; name; T

    function JacobianDict{R,S}(nd::AbstractDict{R,S}; outputs=nothing, inputs=nothing,
                                name=nothing, T=nothing, check=false) where {R,S}
        check && ensure_valid_jacobiandict(nd)
        _out = isnothing(outputs) ? OrderedSet{String}(keys(nd)) : OrderedSet{String}(string(k) for k in outputs)
        _inp = if isnothing(inputs)
            inp = OrderedSet{String}()
            for v in values(nd)
                v isa AbstractDict && (inp = inp ∪ OrderedSet{String}(string(k) for k in keys(v)))
            end
            inp
        else OrderedSet{String}(string(k) for k in inputs) end
        (isempty(_out) || isempty(_inp)) && (_out = OrderedSet{String}(); _inp = OrderedSet{String}())
        _name = isnothing(name) ? "NestedDict" : name
        return new{R,S}(nd, _out, _inp, _name, T)
    end
end

# Coercing outer constructor: always create JacobianDict{String,Any}
# This handles Dict{Any,Any}, Dict{String,Dict{Any,Any}}, JacobianDict, etc.
function JacobianDict(nd::AbstractDict; kwargs...)
    # Normalize to Dict{String,Any} with String-keyed inner dicts
    # Use pairs() to get (key,value) tuples — iterate(NestedDict) yields keys only!
    d = Dict{String,Any}()
    for (k, v) in pairs(nd)
        ks = string(k)
        if v isa AbstractDict
            d[ks] = Dict{String,Any}(string(ik) => iv for (ik, iv) in pairs(v))
        else
            d[ks] = v
        end
    end
    return JacobianDict{String,Any}(d; kwargs...)
end

function jacobian_identity(ks)
    ks_str = OrderedSet{String}(string(k) for k in ks)
    JacobianDict(Dict{String,Any}(k => Dict{String,Any}(k => IdentityMatrix()) for k in ks_str);
                 outputs=ks_str, inputs=ks_str)
end

function addinputs(J::JacobianDict)
    extras = [x for x in J.inputs if x ∉ J.outputs]
    return J ∪ jacobian_identity(extras)
end


# Custom show methods to avoid Base's AbstractDict show (which calls iterate → outputs only)
function Base.show(io::IO, nd::JacobianDict)
    print(io, "JacobianDict{$(eltype(nd.outputs))}(outputs=$(collect(nd.outputs)), inputs=$(collect(nd.inputs)))")
end
function Base.show(io::IO, ::MIME"text/plain", nd::JacobianDict)
    n = length(nd.nesteddict)
    print(io, "JacobianDict{String, Any} with $n entries")
    if n > 0 && n <= 20
        print(io, ":")
        for (o, inner) in pairs(nd.nesteddict)
            print(io, "\n  \"$o\" => {$(join(keys(inner), ", "))}")
        end
    end
end

Base.:*(J::JacobianDict, K::JacobianDict) = compose(J, K)
Base.:*(J::JacobianDict, x::Bijection)    = remap(J, x)
Base.:*(x::Bijection, J::JacobianDict)    = remap(J, x)
Base.:*(J::JacobianDict, x::Any)          = apply(J, x)
Base.isempty(J::JacobianDict) = isempty(J.outputs) || isempty(J.inputs)

function remap(J::JacobianDict, x::Bijection)
    isempty(x) && return J
    nd = x * J.nesteddict
    for o in keys(nd); nd[o] = x * nd[o]; end
    return JacobianDict(nd; inputs=x*J.inputs, outputs=x*J.outputs)
end

function compose(J::JacobianDict, K::JacobianDict)
    (!isnothing(J.T) && !isnothing(K.T) && J.T != K.T) &&
        error("Inconsistent JacobianDict dimensions $(J.T) and $(K.T)")
    o_list = J.outputs
    m_list = Tuple(Set(J.inputs) ∩ Set(K.outputs))
    i_list = K.inputs
    J_oi   = Dict{String,Any}()
    for o in o_list
        o_str = string(o)
        J_oi[o_str] = Dict{String,Any}()
        Jo = J.nesteddict[o_str]  # inner dict: input -> Jacobian
        for i in i_list
            i_str = string(i)
            Jout = nothing
            for m in m_list
                m_str = string(m)
                Jm = K.nesteddict[m_str]  # inner dict
                (haskey(Jo, m_str) && haskey(Jm, i_str)) || continue
                term = Jo[m_str] * Jm[i_str]
                Jout = isnothing(Jout) ? term : Jout + term
            end
            !isnothing(Jout) && (J_oi[o_str][i_str] = Jout)
        end
    end
    return JacobianDict(J_oi; outputs=o_list, inputs=i_list)
end

function apply(J::JacobianDict, x::AbstractDict)
    x = ImpulseDict(x)
    inputs = keys(x) ∩ Set(J.inputs)   # FIX: was `set(J.inputs)` (lowercase)
    y = Dict{String,Any}()
    for o in J.outputs
        o_str = string(o)
        y[o_str] = zeros(x.T)
        Jo = J.nesteddict[o_str]  # inner dict: input -> Jacobian
        for i in inputs
            i_str = string(i)
            haskey(Jo, i_str) && (y[o_str] += Jo[i_str] * x[i_str])
        end
    end
    return x ∪ ImpulseDict(y; T=x.T)   # FIX: was `x.t` (wrong case)
end

function pack(J::JacobianDict; T=nothing)
    T_use = isnothing(T) ? (isnothing(J.T) ? error("Need T to pack") : J.T) : T
    (!isnothing(J.T) && T_use != J.T) && error("Dimension mismatch")
    K = zeros(length(J.outputs)*T_use, length(J.inputs)*T_use)
    for (io,o) in enumerate(J.outputs), (ii,i) in enumerate(J.inputs)
        o_str = string(o); i_str = string(i)
        Jo = J.nesteddict[o_str]  # inner dict
        J_oi = haskey(Jo, i_str) ? Jo[i_str] : nothing
        if !isnothing(J_oi)
            K[T_use*(io-1)+1:T_use*io, T_use*(ii-1)+1:T_use*ii] = make_matrix(J_oi, T_use)
        end
    end
    return K
end

function unpack(bigjac, outputs, inputs, T)
    jd = Dict{String,Any}()
    for (io,o) in enumerate(outputs)
        jd[string(o)] = Dict{String,Any}()
        for (ii,i) in enumerate(inputs)
            jd[string(o)][string(i)] = bigjac[T*(io-1)+1:T*io, T*(ii-1)+1:T*ii]
        end
    end
    return JacobianDict(jd; outputs=outputs, inputs=inputs, T=T)
end

struct FactoredJacobianDict
    H_U_factored; targets; unknowns; T

    function FactoredJacobianDict(jd::JacobianDict; T=nothing)
        T_use = isnothing(T) ? (isnothing(jd.T) ? error("Need T to factor") : jd.T) : T
        length(jd.outputs) != length(jd.inputs) && error("Unequal inputs/outputs")
        H_U = pack(jd; T=T_use)
        return new(lu(H_U), jd.outputs, jd.inputs, T_use)
    end

    # 4-arg constructor used by remap()
    FactoredJacobianDict(H_U_factored, targets, unknowns, T) =
        new(H_U_factored, targets, unknowns, T)
end

Base.show(io::IO, f::FactoredJacobianDict) =
    print(io, "<FactoredJacobianDict unknowns=$(f.unknowns), targets=$(f.targets)>")

Base.:*(f::FactoredJacobianDict, J::JacobianDict) = compose(f, J)
Base.:*(f::FactoredJacobianDict, x::Bijection)    = remap(f, x)
Base.:*(x::Bijection, f::FactoredJacobianDict)    = remap(f, x)
Base.:*(f::FactoredJacobianDict, x::Any)          = apply(f, x)

function remap(f::FactoredJacobianDict, x::Bijection)
    isempty(x) && return f
    return FactoredJacobianDict(f.H_U_factored, x*f.targets, x*f.unknowns, f.T)
end

function compose(f::FactoredJacobianDict, J::JacobianDict)
    Jsub = pack(J[[o for o in f.targets if o in J.outputs]]; T=f.T)
    return unpack(-(f.H_U_factored \ Jsub), f.unknowns, J.inputs, f.T)
end

function apply(f::FactoredJacobianDict, x::Union{ImpulseDict, Dict})
    xsub = pack(get(ImpulseDict(x), f.targets); force_order=f.targets)
    return unpack(-(f.H_U_factored \ xsub), f.unknowns, f.T)
end

function ensure_valid_jacobiandict(d)
    isempty(d) || d isa JacobianDict && return
    first(keys(d)) isa AbstractString || error("Keys must be strings")
    jd = first(values(d))
    jd isa Dict || error("Values must be Dicts")
    isempty(jd) && return
    first(keys(jd)) isa AbstractString || error("Inner keys must be strings")
    jac = first(values(jd))
    jac isa Jacobian || error("Inner values must be Jacobians")
end

function verify_saved_jacobian(name, Js, outputs, inputs, T)
    name ∉ keys(Js) && return false
    J = Js[name]
    J isa JacobianDict || (@warn "Not a JacobianDict"; return false)
    Set(outputs) ⊆ Set(J.outputs) || (@warn "Missing outputs"; return false)
    Set(inputs)  ⊆ Set(J.inputs)  || (@warn "Missing inputs";  return false)
    !isnothing(T) && size(J[first(J.outputs)][first(J.inputs)])[end] != T && (@warn "Wrong T"; return false)
    return true
end
