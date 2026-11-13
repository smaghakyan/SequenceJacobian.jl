# SteadyStateDict which is subtype of ResultDict with same toplevel and internals structure
mutable struct SteadyStateDict{R, S} <: ResultDict{R, S}
    toplevel::AbstractDict{R, S}
    internals

    function SteadyStateDict{R, S}(data::ResultDict{R, S}; internals = nothing) where {R, S}
        isa(internals, Nothing) || error("Supplying $(nameof(typeof(data))) and also internals to constructor not allowed")
        _toplevel = copy(data.toplevel)
        _internals = copy(data.internals)
        return new{R, S}(_toplevel, _internals)
    end

    function SteadyStateDict{R, S}(data::AbstractDict{R, S}; internals = nothing) where {R, S}
        _toplevel = copy(data)
        _internals = internals isa Nothing ? Dict{String,Any}() : copy(internals)
        return new{R, S}(_toplevel, _internals)
    end
end

# Always create SteadyStateDict{String,Any} — coerce keys to String, values to Any
function SteadyStateDict(data::AbstractDict; internals=nothing)
    d = Dict{String,Any}(string(k) => v for (k,v) in data)
    _internals = internals isa Nothing ? Dict{String,Any}() : copy(internals)
    return SteadyStateDict{String,Any}(d; internals=_internals)
end

# New SteadyStateDict after removing certain entries from toplevel
function difference(s::SteadyStateDict, data_to_remove)
    return SteadyStateDict(Dict(setdiff(s.toplevel, data_to_remove)); internals=deepcopy(s.internals))
end

# Finds keys with values > 1
function _vector_valued(s::SteadyStateDict) 
    result = OrderedSet{String}()
    for (k, v) ∈ s.toplevel
        try
            length(v) > 1 && push!(result, k)
        catch
            # If length() fails (unusual type), skip — not vector-valued
        end
    end
    return result
end

UserProvidedSS = Dict{String, Union{Real, AbstractArray}}

# Allow construction from typed sub-dicts like Dict{String,Float64} -> SteadyStateDict{String,Any}
# S must be concrete (not Any) to avoid infinite recursion
function SteadyStateDict{String,Any}(data::AbstractDict{String,S}; internals=nothing) where {S<:Union{Real,AbstractArray}}
    d = Dict{String,Any}(data)
    return SteadyStateDict{String,Any}(d; internals=internals)
end
