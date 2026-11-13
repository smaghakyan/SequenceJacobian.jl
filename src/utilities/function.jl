# Function: ExtendedFunction and CombinedExtendedFunction.
# Julia port of Python SSJ utilities/function.py

using OrderedCollections: OrderedSet, OrderedDict

abstract type AbstractExtendedFunction end

struct ExtendedFunction <: AbstractExtendedFunction
    f::Union{Function, AbstractExtendedFunction}
    name::String
    inputs::OrderedSet{String}
    outputs::OrderedSet{String}

    function ExtendedFunction(f)
        f isa AbstractExtendedFunction && return new(f.f, f.name, f.inputs, f.outputs)
        return new(f, metadata(f)...)
    end

    # Explicit constructor: use when outputs are already known
    function ExtendedFunction(f, name::String, inputs::OrderedSet{String}, outputs::OrderedSet{String})
        _f = f isa AbstractExtendedFunction ? f.f : f
        return new(_f, name, inputs, outputs)
    end
end

_make_tuple(x) = (x isa Union{Tuple, AbstractArray}) ? x : (x,)

function input_defaults(f::Function)
    return Dict{String,Any}()
end

# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------
function input_list(f::Function)
    m = _last_method(f)
    params = Base.method_argnames(m)[2:end]
    kwargs = Base.kwarg_decl(m)
    all_args = filter(!isempty ∘ string, vcat(params, kwargs))
    return OrderedSet{String}(string.(all_args))
end

# ---------------------------------------------------------------------------
# Output detection
#
# Julia cannot report the names in a function's returned NamedTuple without
# running it, so the names are recovered in three stages:
#   1. Parse the function's own source with Julia's parser and read the names
#      off the last `return` (or the final expression). Working on the syntax
#      tree instead of on text means nested parentheses, multi-line returns,
#      comments, docstrings, one-line `for ... end` loops and `do` blocks no
#      longer break detection.
#   2. Call the function on placeholder inputs and read the NamedTuple keys.
#   3. Give up and return the empty set (callers may set `.outputs` by hand).
# ---------------------------------------------------------------------------
function output_list(f::Function)
    names, certain = _output_names_from_source(f)
    (!isempty(names) && certain) && return names

    called = _output_names_from_call(f)
    !isempty(called) && return called

    # `return x` with a bare name: use it only if calling could not tell us more
    return names
end

# Method used for introspection. `methods(f)` is not indexable on old Julia
# versions, hence the explicit iteration.
function _last_method(f::Function)
    m = nothing
    for mm in methods(f); m = mm; end
    return m
end

# Parse the definition of `f` out of its source file. Returns an Expr or nothing.
function _function_expr_from_source(f::Function)
    m = _last_method(f)
    isnothing(m) && return nothing
    file = replace(String(m.file), "\\" => "/")
    actual_file = ""
    for candidate in (file, replace(file, "/" => "\\"), String(m.file))
        isfile(candidate) && (actual_file = candidate; break)
    end
    isempty(actual_file) && return nothing

    src   = read(actual_file, String)
    lines = split(src, '\n')
    fname = String(nameof(f))
    startswith(fname, "#") && return nothing      # anonymous function
    qname = replace(fname, r"([\\.^$|?*+()\[\]{}!])" => s"\\\1")
    long  = Regex("^\\s*(?:@[\\w.]+\\s+)*function\\s+(?:[\\w.]+\\.)?" * qname * "\\s*[({]")
    short = Regex("^\\s*(?:@[\\w.]+\\s+)*(?:[\\w.]+\\.)?" * qname * "\\s*\\(.*\\)\\s*(?:where\\s+[^=]+)?=[^=]")

    # `method.line` may point at the signature, at the first body line, or at a
    # docstring, depending on the Julia version: search outward in both directions.
    start = 0
    for off in 0:40
        for idx in (Int(m.line) - off, Int(m.line) + off)
            (1 <= idx <= length(lines)) || continue
            if occursin(long, lines[idx]) || occursin(short, lines[idx])
                start = idx; break
            end
        end
        start != 0 && break
    end
    start == 0 && return nothing

    offset = 1
    for i in 1:start-1
        offset += ncodeunits(lines[i]) + 1
    end
    ex = try
        first(Meta.parse(src, offset; raise=false))
    catch
        nothing
    end
    return ex
end

# Strip macro calls / docstrings wrapped around a definition.
function _unwrap_definition(ex)
    while ex isa Expr && ex.head == :macrocall
        ex = ex.args[end]
    end
    return ex
end

# Collect `return` expressions in source order, without descending into nested
# function definitions (their returns belong to the inner function).
function _collect_returns!(acc::Vector{Any}, ex)
    ex isa Expr || return acc
    if ex.head == :return
        push!(acc, isempty(ex.args) ? nothing : ex.args[1])
        return acc
    end
    (ex.head == :function || ex.head == :(->)) && return acc
    if ex.head == :(=) && ex.args[1] isa Expr && ex.args[1].head in (:call, :where)
        return acc                                  # short-form inner function
    end
    for a in ex.args
        _collect_returns!(acc, a)
    end
    return acc
end

# Last "real" expression of a block (ignoring line-number nodes).
function _last_expression(ex)
    while ex isa Expr && ex.head == :block
        idx = findlast(a -> !(a isa LineNumberNode), ex.args)
        isnothing(idx) && return nothing
        ex = ex.args[idx]
    end
    return ex
end

# Names carried by a returned expression. Second value: are we sure these are
# output names (a named or plain tuple), or is it just a bare variable name?
function _names_of_returned(ex)
    names = String[]
    if ex isa Symbol
        return [string(ex)], false
    elseif ex isa Expr && ex.head == :tuple
        for a in ex.args
            if a isa Symbol
                push!(names, string(a))
            elseif a isa Expr && (a.head == :(=) || a.head == :kw) && a.args[1] isa Symbol
                push!(names, string(a.args[1]))
            elseif a isa Expr && a.head == :parameters
                for q in a.args
                    if q isa Symbol
                        push!(names, string(q))
                    elseif q isa Expr && (q.head == :kw || q.head == :(=)) && q.args[1] isa Symbol
                        push!(names, string(q.args[1]))
                    elseif q isa Expr && q.head == :. && q.args[end] isa QuoteNode
                        push!(names, string(q.args[end].value))
                    else
                        return String[], false
                    end
                end
            else
                return String[], false
            end
        end
        return names, true
    elseif ex isa Expr && ex.head == :(=) && ex.args[1] isa Symbol
        # `name(args) = (x = expr)` style single output
        return [string(ex.args[1])], true
    end
    return names, false
end

function _output_names_from_source(f::Function)
    try
        ex = _unwrap_definition(_function_expr_from_source(f))
        (ex isa Expr && length(ex.args) >= 2) || return OrderedSet{String}(), false
        body = ex.args[end]
        rets = _collect_returns!(Any[], body)
        candidate = isempty(rets) ? _last_expression(body) : rets[end]
        names, certain = _names_of_returned(candidate)
        return OrderedSet{String}(names), certain
    catch
        return OrderedSet{String}(), false
    end
end

function _output_names_from_call(f::Function)
    m = _last_method(f)
    isnothing(m) && return OrderedSet{String}()
    kwargs = Base.kwarg_decl(m)
    isempty(kwargs) && return OrderedSet{String}()

    # Placeholders: first the block-algebra scalar (simple-block functions are
    # written against it), then plain numbers and small arrays.
    for dummy_val in Any[_probe_scalar(), 1.0, ones(2), ones(2, 2), ones(3)]
        isnothing(dummy_val) && continue
        dummy_kw = Dict(k => dummy_val for k in kwargs)
        try
            result = f(; dummy_kw...)
            result isa NamedTuple && return OrderedSet{String}(string.(keys(result)))
        catch
        end
    end
    return OrderedSet{String}()
end

# `ignore` is defined later in the include order (simple_displacement.jl);
# resolve it at call time.
_probe_scalar() = isdefined(@__MODULE__, :ignore) ? Base.invokelatest(getfield(@__MODULE__, :ignore), 1.0) : nothing

function metadata(f::Function)
    name = String(nameof(f))
    name = (length(name) > 0 && name[1] == '_') ? name[2:end] : name
    inputs  = input_list(f)
    outputs = output_list(f)
    return name, inputs, outputs
end

Base.show(io::IO, f::AbstractExtendedFunction) =
    print(io, "<$(typeof(f))($(f.name)): [$(join(f.inputs,", "))] -> [$(join(f.outputs,", "))]>")

function (f::AbstractExtendedFunction)(input_dict)
    kw = Dict(Symbol(k) => v for (k,v) in input_dict if k in f.inputs)
    result = f.f(; kw...)
    if result isa NamedTuple
        return Dict(string(k) => v for (k,v) in pairs(result))
    else
        return Dict(zip(f.outputs, Tuple(result)))
    end
end

function wrapped_call(f::AbstractExtendedFunction, input_dict;
                      preprocess=nothing, postprocess=nothing)
    kw = if !isnothing(preprocess)
        Dict(Symbol(k) => preprocess(v) for (k,v) in input_dict if k in f.inputs)
    else
        Dict(Symbol(k) => v for (k,v) in input_dict if k in f.inputs)
    end
    result = f.f(; kw...)
    output_dict = if result isa NamedTuple
        Dict(string(k) => v for (k,v) in pairs(result))
    else
        Dict(zip(f.outputs, Tuple(result)))
    end
    !isnothing(postprocess) && (output_dict = Dict(k => postprocess(v) for (k,v) in output_dict))
    return output_dict
end

function differentiable(f::AbstractExtendedFunction, input_dict; h=1e-4, twosided=false)
    return DifferentiableExtendedFunction(f.f, f.name, f.inputs, f.outputs, input_dict; h=h, twosided=twosided)
end

hide_zero_values(d) = Dict(k => v for (k,v) in d
    if !all(x -> isapprox(x...; atol=1e-8, rtol=1e-5), zip(zeros(length(v)), v)))

mutable struct DifferentiableExtendedFunction <: AbstractExtendedFunction
    f; name; inputs; outputs
    input_dict; output_dict; h; default_twosided

    function DifferentiableExtendedFunction(f, name, inputs, outputs, input_dict; h=1e-4, twosided=false)
        return new(f, name, inputs, outputs, input_dict, nothing, h, twosided)
    end
end

function diff(f::DifferentiableExtendedFunction, shock_dict; h=nothing, hide_zeros=false, twosided=nothing)
    isnothing(twosided) && (twosided = f.default_twosided)
    return twosided ? diff2(f, shock_dict; h=h, hide_zeros=hide_zeros) :
                      diff1(f, shock_dict; h=h, hide_zeros=hide_zeros)
end

function diff1(f::DifferentiableExtendedFunction, shock_dict; h=nothing, hide_zeros=false)
    isnothing(h) && (h = f.h)
    isnothing(f.output_dict) && (f.output_dict = f(f.input_dict))
    shocked = Dict{String,Any}(f.input_dict...,
        (k => f.input_dict[k] .+ h .* shock for (k,shock) in shock_dict if k in keys(f.input_dict))...)
    shocked_out = f(shocked)
    deriv = Dict{String,Any}(k => (shocked_out[k] - f.output_dict[k]) / h for k in keys(f.output_dict))
    hide_zeros && (deriv = hide_zero_values(deriv))
    return deriv
end

function diff2(f::DifferentiableExtendedFunction, shock_dict; h=nothing, hide_zeros=false)
    isnothing(h) && (h = f.h)
    up = Dict{String,Any}(f.input_dict...,
        (k => f.input_dict[k] .+ h .* shock for (k,shock) in shock_dict if k in keys(f.input_dict))...)
    dn = Dict(f.input_dict...,
        (k => f.input_dict[k] .- h .* shock for (k,shock) in shock_dict if k in keys(f.input_dict))...)
    out_up = f(up); out_dn = f(dn)
    deriv = Dict{String,Any}(k => (out_up[k] - out_dn[k]) / (2h) for k in keys(out_dn))
    hide_zeros && (deriv = hide_zero_values(deriv))
    return deriv
end

abstract type AbstractCombinedExtendedFunction <: AbstractExtendedFunction end

struct CombinedExtendedFunction <: AbstractCombinedExtendedFunction
    dag; inputs; outputs; functions; name

    function CombinedExtendedFunction(fs; name=nothing)
        _dag  = DAG([ExtendedFunction(f) for f in fs])
        _fns  = OrderedDict(b.name => b for b in _dag.blocks)
        _name = isnothing(name) ? begin
            ns = collect(keys(_fns))
            length(ns) == 1 ? ns[1] : "$(ns[1])_$(ns[end])"
        end : name
        return new(_dag, _dag.inputs, _dag.outputs, _fns, _name)
    end
end

function (f::AbstractCombinedExtendedFunction)(input_dict; outputs=nothing)
    fns = collect(values(f.functions))
    !isnothing(outputs) && (fns = [fns[i] for i in visit_from_outputs(f.dag, outputs)])
    results = copy(input_dict)
    for g in fns; merge!(results, g(results)); end
    isnothing(outputs) ? results : Dict{String,Any}(string(k) => results[string(k)] for k in outputs)
end

function _filter(f::AbstractCombinedExtendedFunction, fn_list, inputs; outputs=nothing)
    nums = visit_from_inputs(f.dag, inputs)
    !isnothing(outputs) && (nums = nums ∩ visit_from_outputs(f.dag, outputs))
    return [fn_list[n] for n in nums]
end

function call_on_deviations(f::AbstractCombinedExtendedFunction, ss, dev_dict; outputs=nothing)
    fns = _filter(f, collect(values(f.functions)), dev_dict; outputs=outputs)
    results = Dict{String,Any}(); input_dict = Dict{String,Any}(ss..., dev_dict...)
    for g in fns
        out = g(input_dict); merge!(results, out); merge!(input_dict, out)
    end
    isnothing(outputs) ? results : Dict{String,Any}(k=>v for (k,v) in results if k in outputs)
end

add(f::AbstractCombinedExtendedFunction, g::Union{Function,AbstractExtendedFunction}) =
    CombinedExtendedFunction([collect(values(f.functions)); g])
add(f::AbstractCombinedExtendedFunction, g) =
    CombinedExtendedFunction([collect(values(f.functions))..., collect(g)...])
remove(f::AbstractCombinedExtendedFunction, name::AbstractString) =
    CombinedExtendedFunction([v for (k,v) in f.functions if k != name])
remove(f::AbstractCombinedExtendedFunction, names) =
    CombinedExtendedFunction([v for (k,v) in f.functions if k ∉ names])

function differentiable(f::AbstractCombinedExtendedFunction, input_dict; h=1e-5, twosided=false)
    return DifferentiableCombinedExtendedFunction(f.functions, f.dag, f.name,
                                                   f.inputs, f.outputs, input_dict; h=h, twosided=twosided)
end

struct DifferentiableCombinedExtendedFunction <: AbstractCombinedExtendedFunction
    dag; name; inputs; outputs; diff_functions; default_twosided

    function DifferentiableCombinedExtendedFunction(functions, dag, name, inputs, outputs, input_dict; h=1e-5, twosided=false)
        diff_fns = OrderedDict(k => differentiable(f, input_dict; h=h, twosided=twosided) for (k,f) in functions)
        return new(dag, name, inputs, outputs, diff_fns, twosided)
    end
end

# `twosided=nothing` means "use the rule this object was built with" (the HetBlock
# builds its hetinput/hetoutput functions with the block's own `twosided` setting).
function diff(f::DifferentiableCombinedExtendedFunction, shock_dict; h=nothing, outputs=nothing, hide_zeros=false, twosided=nothing)
    isnothing(twosided) && (twosided = f.default_twosided)
    return twosided ? diff2(f, shock_dict; h=h, outputs=outputs, hide_zeros=hide_zeros) :
                      diff1(f, shock_dict; h=h, outputs=outputs, hide_zeros=hide_zeros)
end

function diff1(f::DifferentiableCombinedExtendedFunction, shock_dict; h=nothing, outputs=nothing, hide_zeros=false)
    fns = _filter(f, collect(values(f.diff_functions)), shock_dict; outputs=outputs)
    shock = copy(shock_dict); results = Dict{String,Any}()
    for g in fns
        out = diff1(g, shock; h=h, hide_zeros=hide_zeros)
        merge!(results, out); merge!(shock, out)
    end
    isnothing(outputs) ? results : Dict{String,Any}(k=>v for (k,v) in results if k in outputs)
end

function diff2(f::DifferentiableCombinedExtendedFunction, shock_dict; h=nothing, outputs=nothing, hide_zeros=false)
    fns = _filter(f, collect(values(f.diff_functions)), shock_dict; outputs=outputs)
    shock = copy(shock_dict); results = Dict{String,Any}()
    for g in fns
        out = diff2(g, shock; h=h, hide_zeros=hide_zeros)
        merge!(results, out); merge!(shock, out)
    end
    isnothing(outputs) ? results : Dict{String,Any}(k=>v for (k,v) in results if k in outputs)
end
