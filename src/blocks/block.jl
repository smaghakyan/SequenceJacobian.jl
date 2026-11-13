# Abstract base class for all general Block objects
abstract type Block end

# Python @property equivalents for inputs and outputs
get_inputs(b)  = b.inputs
get_outputs(b) = b.outputs

# =============================================================================
# Options
#
# Every block carries default options for each method (`steady_state_options`,
# `jacobian_options`, ...). They can be overridden, in increasing priority, by
#   * the block's own dictionary      hh.steady_state_options["backward_tol"] = 1e-10
#   * a per-block entry in `options`  options = Dict("hh" => Dict("backward_tol" => 1e-10))
#   * keyword arguments               steady_state(model, cal; backward_tol = 1e-10)
# Keyword arguments travel down through combined and solved blocks, so they reach
# every block that knows the option. Option names may be Strings or Symbols.
# =============================================================================
function get_options(b::Block, options, kwargs, method)
    sym = Symbol(method * "_options")
    own = hasproperty(b, sym) ? getproperty(b, sym) : Dict{String,Any}()
    merged = Dict{String,Any}(string(k) => v for (k, v) ∈ own)
    names = collect(keys(merged))

    if options isa AbstractDict && haskey(options, b.name) && options[b.name] isa AbstractDict
        for (k, v) ∈ options[b.name]
            merged[string(k)] = v
        end
    end
    if !(kwargs isa Tuple)
        for (k, v) ∈ kwargs
            merged[string(k)] = v
        end
    end
    return Dict{String,Any}(k => merged[k] for k ∈ names)
end

# String-keyed option dictionaries cannot be splatted as keyword arguments.
_kw(d) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) ∈ d)

# Calls between the generic entry points below and the block-specific methods are
# dynamic anyway (blocks are stored untyped). The impulse functions call each other
# recursively through combined and solved blocks; sending those calls through this
# barrier stops the compiler from trying to infer the whole recursive call graph at
# once (which could make the first call take minutes to compile).
const _dyn = Base.invokelatest

# =============================================================================
# Steady state
# =============================================================================

# Evaluates a partial equilibrium steady state of Block given calibration params
function steady_state(b::Block, calibration; dissolve=[], options=Dict(), kwargs...)
    inputs = OrderedSet{String}(string(k) for k ∈ b.inputs)

    if b isa ParentBlock
        for k in dissolve
            inputs = inputs ∪ OrderedSet{String}(string(x) for x ∈ keys(get_attribute(b, k, :unknowns)))
        end
    end

    calibration = SteadyStateDict(calibration)[inputs]

    if b isa ParentBlock
        return b.M * _steady_state(b, inv(b.M) * calibration; dissolve=dissolve, options=options, kwargs...)
    else
        own_options = get_options(b, options, kwargs, "steady_state")
        return b.M * _steady_state(b, inv(b.M) * calibration; _kw(own_options)...)
    end
end

# Evaluates a general equilibrium steady state of Block given calibration, a set of
# unknowns (endogenous variables to be solved for) and targets (conditions that must
# hold in general equilibrium).
function solve_steady_state(b::Block, calibration, unknowns, targets; dissolve=[], options=Dict(), kwargs...)
    # NB: `opts` are this block's solver settings; `options` stays the per-block
    # dictionary and is what gets passed down (it used to be overwritten here, so
    # per-block options never reached the blocks inside a solve).
    opts   = get_options(b, options, kwargs, "solve_steady_state")
    ss     = SteadyStateDict(calibration)
    solver = !isempty(opts["solver"]) ? opts["solver"] : provide_solver_default(unknowns)
    unknown_names = [string(k) for k ∈ keys(unknowns)]     # one fixed order, used throughout
    last_residual = Ref{Any}(nothing)

    # Accepts scalar (1D root finders) or vector (multi-dimensional solvers)
    function residual(unknown_values)
        vals = unknown_values isa Number ? [Float64(unknown_values)] : collect(Float64, unknown_values)
        for (k, v) ∈ zip(unknown_names, vals)
            ss[k] = v
        end
        merge!(ss, steady_state(b, ss; dissolve=dissolve, options=options, kwargs...))
        tv = compute_target_values(targets, ss)
        last_residual[] = tv
        if unknown_values isa Number
            return tv isa AbstractArray ? tv[1] : Float64(tv)
        else
            return tv isa Number ? [tv] : collect(tv)
        end
    end

    sol_dict = solve_for_unknowns(residual, unknowns, solver,
                                  get(opts, "solver_kwargs", Dict());
                                  tol      = get(opts, "ttol", 2e-12),
                                  verbose  = get(opts, "verbose", false),
                                  constrained_method  = get(opts, "constrained_method", "linear_continuation"),
                                  constrained_kwargs  = get(opts, "constrained_kwargs",  Dict()))

    # One final evaluation at the solution (in the same order the solver used)
    residual([sol_dict[k] for k ∈ keys(unknowns)])

    # A solver returning is not the same as the targets being hit.
    final = last_residual[]
    worst = final isa Number ? abs(final) : maximum(abs, final)
    if !(worst < max(1e-5, 1e3 * get(opts, "ttol", 2e-12)))
        @warn "solve_steady_state('$(b.name)'): largest target residual at the returned point is $(worst). The steady state has probably NOT converged."
    end
    return ss
end

# =============================================================================
# Linear impulse responses
# =============================================================================

# Partial equilibrium linear impulse responses to the paths in `inputs`.
function impulse_linear(b::Block, ss, inputs::ImpulseDict, outputs=nothing, Js=Dict(), options=Dict(); kwargs...)
    own_options = get_options(b, options, kwargs, "impulse_linear")
    input_names = make_ordered_set(inputs)
    actual_outputs, inputs_as_outputs = process_outputs(b, ss, input_names, make_ordered_set(outputs))

    if isempty(actual_outputs)
        return inputs[inputs_as_outputs]
    end

    # Reuse an already computed Jacobian of this block when there is one.
    if !(b isa ParentBlock) && Js isa AbstractDict && haskey(Js, b.name) && Js[b.name] isa JacobianDict &&
            (input_names ∩ b.inputs) ⊆ Js[b.name].inputs && actual_outputs ⊆ Js[b.name].outputs
        out = apply(Js[b.name][actual_outputs, input_names ∩ b.inputs], inputs)[actual_outputs]
    elseif b isa ParentBlock
        out = b.M * _dyn(_impulse_linear, b, inv(b.M) * ss, inv(b.M) * inputs, inv(b.M) * actual_outputs, Js, options)
    else
        out = b.M * _dyn(_impulse_linear, b, inv(b.M) * ss, inv(b.M) * inputs, inv(b.M) * actual_outputs, Js; _kw(own_options)...)
    end
    return inputs[inputs_as_outputs] ∪ out
end

# Plain dictionaries of paths are accepted as well.
impulse_linear(b::Block, ss, inputs::Dict, outputs=nothing; kwargs...) =
    impulse_linear(b, ss, ImpulseDict(inputs), outputs; kwargs...)

# General equilibrium linear impulse responses given unknowns and targets.
function solve_impulse_linear(b::Block, ss, unknowns, targets, inputs, outputs=nothing;
                              Js=Dict(), options=Dict(), H_U_factored=nothing, kwargs...)
    inputs      = inputs isa ImpulseDict ? inputs : ImpulseDict(inputs)
    unknowns_os = make_ordered_set(unknowns)
    targets_os  = make_ordered_set(targets)
    input_names = make_ordered_set(inputs)
    actual_outputs, inputs_as_outputs = process_outputs(b, ss, input_names ∪ unknowns_os, make_ordered_set(outputs))
    T = inputs.T

    Js = _dyn(partial_jacobians, b, ss, input_names ∪ unknowns_os, setdiff(actual_outputs ∪ targets_os, unknowns_os);
              T=T, Js=Js, options=options, kwargs...)

    # effect of the shocks on the targets, holding the unknowns fixed
    dH = get(_dyn(impulse_linear, b, ss, inputs, targets_os, Js, options; kwargs...), targets_os)

    if isnothing(H_U_factored)
        H_U = pack(_dyn(jacobian, b, ss, unknowns_os, targets_os; T=T, Js=Js, options=options, kwargs...); T=T)
        dU  = unpack(-(H_U \ pack(dH; force_order=targets_os)), unknowns_os, T)
    else
        dU = apply(H_U_factored, dH)
    end

    # responses of everything else to shocks and unknowns together
    rest = _dyn(impulse_linear, b, ss, dU ∪ inputs, setdiff(actual_outputs, unknowns_os), Js, options; kwargs...)
    return (inputs ∪ dU)[inputs_as_outputs] ∪ rest
end

# =============================================================================
# Jacobians
# =============================================================================

function partial_jacobians(b::Block, ss, inputs = nothing, outputs = nothing; T = nothing, Js = Dict(), options = Dict(), kwargs...)
    if inputs isa Nothing
        inputs = b.inputs
    end
    if outputs isa Nothing
        outputs = b.outputs
    end
    # Always ensure String-typed sets to avoid Bijection dispatch failures
    inputs  = make_ordered_set(inputs)
    outputs = make_ordered_set(outputs)

    if (b.name ∈ keys(Js)) && (Js[b.name] isa JacobianDict) && (inputs ⊆ Js[b.name].inputs) && (outputs ⊆ Js[b.name].outputs)
        return Dict{String,Any}(b.name => Js[b.name][outputs, inputs])
    end

    if !(b isa ParentBlock)
        jac = jacobian(b, ss, inputs, outputs; T=T, options=options, kwargs...)
        return !isempty(jac) ? Dict{String,Any}(b.name => jac) : Dict{String,Any}()
    end

    own_options = get_options(b, options, kwargs, "partial_jacobians")

    partial = _partial_jacobians(b, inv(b.M) * ss, inv(b.M) * inputs, inv(b.M) * outputs; T=T, Js=Js, options=options, _kw(own_options)...)

    if b.name ∈ keys(partial)
        partial[b.name] = b.M * partial[b.name]
    end
    return partial
end

# Calculate a partial equilibrium Jacobian to a set of input shocks at steady state ss
function jacobian(b::Block, ss, inputs, outputs = nothing; T = nothing, Js = Dict(), options = Dict(), kwargs...)
    own_options = get_options(b, options, kwargs, "jacobian")
    inputs  = make_ordered_set(inputs)
    outputs = make_ordered_set(outputs)
    outputs, _ = process_outputs(b, ss, Dict{String,Any}(), outputs)

    if b.name ∈ keys(Js) && Js[b.name] isa JacobianDict && inputs ⊆ Js[b.name].inputs && outputs ⊆ Js[b.name].outputs
        return Js[b.name][outputs, inputs]
    end

    if !(b isa ParentBlock)
        return b.M * _jacobian(b, inv(b.M) * ss, inv(b.M) * inputs, inv(b.M) * outputs; T=T, _kw(own_options)...)
    end

    Js = copy(Js)
    if b.name ∈ keys(Js)
        Js[b.name] = inv(b.M) * Js[b.name]
    end
    return b.M * _jacobian(b, inv(b.M) * ss, inv(b.M) * inputs, inv(b.M) * outputs; T=T, Js=Js, options=options, _kw(own_options)...)
end

Base.union(::Nothing, x::Any) = x
Base.union(x::Any, ::Nothing) = x

# Calculate a general equilibrium Jacobian to a set of exogenous shocks
function solve_jacobian(b::Block, ss, unknowns, targets, inputs, outputs = nothing; T=300, Js = Dict(), options = Dict(), H_U_factored = nothing, kwargs...)
    inputs, unknowns = make_ordered_set(inputs), make_ordered_set(unknowns)
    actual_outputs, unknowns_as_outputs = process_outputs(b, ss, unknowns, make_ordered_set(outputs))

    Js = partial_jacobians(b, ss, make_ordered_set(inputs ∪ unknowns), make_ordered_set(setdiff(actual_outputs ∪ make_ordered_set(targets), unknowns)); T=T, Js=Js, options=options, kwargs...)

    H_Z = jacobian(b, ss, inputs, targets; T=T, Js=Js, options=options, kwargs...)

    if H_U_factored isa Nothing
        H_U = pack(jacobian(b, ss, unknowns, targets; T=T, Js=Js, options=options, kwargs...); T=T)
        H_Z_packed = pack(H_Z; T=T)
        U_Z = unpack(- H_U \ H_Z_packed, unknowns, inputs, T)
    else
        U_Z = H_U_factored * H_Z
    end

    self_with_unknowns = combine([U_Z, b])

    return jacobian(self_with_unknowns, ss, inputs, unknowns_as_outputs ∪ actual_outputs; T=T, Js=Js, options=options, kwargs...)
end

# =============================================================================
# Nonlinear impulse responses (perfect-foresight transitions)
# =============================================================================

# Partial equilibrium: feed the input paths through the block. `internals` selects
# individual-level objects to report, either as a list of block names (report
# everything for those blocks) or as Dict(block name => list of names), e.g.
# `internals = Dict("hh" => ["D", "a"])`.
function impulse_nonlinear(b::Block, ss, inputs; outputs = nothing, internals = Dict(), Js = Dict(), options = Dict(), ss_initial = nothing, kwargs...)
    own_options = get_options(b, options, kwargs, "impulse_nonlinear")
    inputs = inputs isa ImpulseDict ? inputs : ImpulseDict(inputs)
    actual_outputs, inputs_as_outputs = process_outputs(b, ss, make_ordered_set(inputs), make_ordered_set(outputs))
    ssi = isnothing(ss_initial) ? nothing : inv(b.M) * ss_initial

    if b isa ParentBlock
        out = b.M * _dyn(_impulse_nonlinear, b, inv(b.M) * ss, inv(b.M) * inputs; outputs = inv(b.M) * actual_outputs, internals=internals, Js=Js, options=options, ss_initial=ssi, _kw(own_options)...)
    elseif hasproperty(b, :internals)
        out = b.M * _dyn(_impulse_nonlinear, b, inv(b.M) * ss, inv(b.M) * inputs; outputs = inv(b.M) * actual_outputs, internals = internals_to_report(b, internals), ss_initial=ssi, _kw(own_options)...)
    else
        out = b.M * _dyn(_impulse_nonlinear, b, inv(b.M) * ss, inv(b.M) * inputs; outputs = inv(b.M) * actual_outputs, ss_initial=ssi, _kw(own_options)...)
    end
    return inputs[inputs_as_outputs] ∪ out
end

# General equilibrium nonlinear impulse responses to the shocks in `inputs`, given
# unknowns and targets.
#
# Quasi-Newton iteration on the stacked path of unknowns, using the steady-state
# Jacobian H_U throughout:
#       U(k+1) = U(k) - λ_k H_U^{-1} H(U(k), Z)
# with a backtracking step length λ_k ∈ {1, 1/2, 1/4, ...} that is accepted only if
# the largest target residual falls. Options (keywords or block options):
#   tol, maxit, verbose   as before
#   backtrack = true      set false for plain full steps (the Python toolbox's rule)
#   max_backtracks = 25
#   warm_start = false    start from the linear solution instead of from zero
function solve_impulse_nonlinear(b::Block, ss, unknowns, targets, inputs; outputs = nothing, internals = Dict(), Js = Dict(), options = Dict(), H_U_factored = nothing, ss_initial = nothing, kwargs...)
    inputs  = inputs isa ImpulseDict ? inputs : ImpulseDict(inputs)
    unknowns, targets = make_ordered_set(unknowns), make_ordered_set(targets)

    input_names = make_ordered_set(inputs)
    actual_outputs, inputs_as_outputs = process_outputs(b, ss, union(input_names, unknowns), make_ordered_set(outputs))

    T = inputs.T

    Js = _dyn(partial_jacobians, b, ss, union(input_names, unknowns), setdiff(union(actual_outputs, targets), unknowns); T=T, Js=Js, options=options, kwargs...)

    if isnothing(H_U_factored)
        H_U  = _dyn(jacobian, b, ss, unknowns, targets; T=T, Js=Js, options=options, kwargs...)
        H_U_factored = FactoredJacobianDict(H_U; T=T)
    end

    opts = get_options(b, options, kwargs, "solve_impulse_nonlinear")
    tol            = opts["tol"]
    maxit          = opts["maxit"]
    verbose        = opts["verbose"]
    backtrack      = get(opts, "backtrack", true)
    max_backtracks = get(opts, "max_backtracks", 25)
    warm_start     = get(opts, "warm_start", false)

    evaluate(U) = _dyn(impulse_nonlinear, b, ss, inputs ∪ U; outputs = actual_outputs ∪ targets, internals=internals, Js=Js, options=options, ss_initial=ss_initial, kwargs...)
    target_errors(res) = Dict{String,Float64}(k => maximum(abs, get(res, k)) for k ∈ targets)
    worst(errs) = maximum(values(errs))

    # Starting point: zero deviation, or the first-order solution
    if warm_start
        dH = get(_dyn(impulse_linear, b, ss, inputs, targets, Js, options; kwargs...), targets)
        U  = apply(H_U_factored, dH)
    else
        U = ImpulseDict(Dict{String,Any}(k => zeros(T) for k ∈ unknowns); T=T)
    end

    verbose && println("Solving '$(b.name)' for $(collect(unknowns)) to hit $(collect(targets))")

    results = evaluate(U)
    errors  = target_errors(results)

    for it ∈ 0:maxit
        if verbose
            println("On iteration $it")
            for k ∈ targets
                println("   max error for $k is $(errors[k])")
            end
        end
        err = worst(errors)
        isfinite(err) || error("Nonlinear transition of '$(b.name)' produced non-finite target residuals.")
        if err < tol
            return (inputs ∪ U)[inputs_as_outputs] ∪ results
        end
        it == maxit && break

        dU = apply(H_U_factored, get(results, targets))
        step = 1.0
        accepted = false
        for bt ∈ 0:max_backtracks
            U_try = U + step * dU
            res_try = try
                evaluate(U_try)
            catch e
                e isa InterruptException && rethrow()
                backtrack || rethrow()
                nothing
            end
            if !isnothing(res_try)
                errs_try = target_errors(res_try)
                e_try = worst(errs_try)
                if !backtrack || (isfinite(e_try) && e_try < err)
                    U, results, errors = U_try, res_try, errs_try
                    accepted = true
                    break
                end
            end
            verbose && println("   backtracking: step length $(step) did not reduce the residual")
            step *= 0.5
        end
        accepted || error("Nonlinear solver for '$(b.name)' could not find an improving step (largest residual $(err)).")
    end
    error("No convergence of '$(b.name)' after $(maxit) quasi-Newton iterations (largest residual $(worst(errors))).")
end

# =============================================================================
# Block manipulation
# =============================================================================

function solved(b::Block, unknowns, targets; name=nothing, solver=nothing, solver_kwargs=nothing)
    isnothing(name) && (name = b.name * "_solved")
    return SolvedBlock(b, name, unknowns, targets; solver=solver, solver_kwargs=isnothing(solver_kwargs) ? Dict() : solver_kwargs)
end

function remap(b::Block, map)
    other = deepcopy(b)
    other.M = b.M * Bijection(map)
    other.inputs = other.M * b.inputs
    other.outputs = other.M * b.outputs
    if hasproperty(b, :input_list)
        other.input_list = other.M * b.input_list
    end
    if hasproperty(b, :output_list)
        other.output_list = other.M * b.output_list
    end
    if hasproperty(b, :non_back_iter_outputs)
        other.non_back_iter_outputs = other.M * b.non_back_iter_outputs
    end
    return other
end

function rename(b::Block; name=nothing, suffix=nothing)
    (isnothing(name) && isnothing(suffix)) && error("rename needs either `name` or `suffix`")
    if b isa ParentBlock
        other = deepcopy(b)
        other.name = isnothing(suffix) ? name : b.name * suffix
        if hasproperty(b, :blocks)
            # only a suffix can be pushed down to the kids (a single name would collide)
            if !isnothing(suffix)
                other.blocks = [rename(block; suffix=suffix) for block ∈ b.blocks]
            end
            other.name, other.kids, other.descendants = construct_parent(other.blocks; name=other.name)
        elseif hasproperty(b, :block)
            if !isnothing(suffix)
                other.block = rename(b.block; suffix=suffix)
            end
            other.name, other.kids, other.descendants = construct_parent([other.block]; name=other.name)
        end
        return other
    else
        if isnothing(suffix)
            return rename_top(b, name)
        else
            return rename_top(b, b.name * suffix)
        end
    end
end

function rename_top(b::Block, name)
    other = deepcopy(b)
    other.name = name
    return other
end

function default_inputs_outputs(b::Block, ss, inputs, outputs)
    if inputs isa Nothing
        inputs = b.inputs
    end
    if outputs isa Nothing
        outputs = setdiff(b.outputs, _vector_valued(ss))
    end
    return OrderedSet{String}(string(k) for k in inputs), OrderedSet{String}(string(k) for k in outputs)
end

function process_outputs(b::Block, ss, inputs, outputs)
    if outputs isa Nothing
        actual_outputs = setdiff(b.outputs, _vector_valued(ss))
        inputs_as_outputs = inputs
    else
        actual_outputs = outputs ∩ b.outputs
        inputs_as_outputs = outputs ∩ inputs
    end

    return actual_outputs, inputs_as_outputs
end

macro add_options(method)
    return :()
end

function input_defaults_smart(b::Block, methodname)
    method = hasproperty(b, methodname) ? getproperty(b, methodname) : nothing
    if method isa Nothing
        return Dict{String,Any}()
    else
        return input_defaults(method)
    end
end

# Which internals should block `b` report? `internals` is either a collection of
# block names (report everything) or a Dict(block name => names).
function internals_to_report(b::Block, internals)
    if internals isa AbstractDict
        return haskey(internals, b.name) ? internals[b.name] : String[]
    elseif isnothing(internals)
        return String[]
    elseif internals isa AbstractString
        return internals == b.name ? b.internals : String[]
    else
        return (b.name in internals) ? b.internals : String[]
    end
end
