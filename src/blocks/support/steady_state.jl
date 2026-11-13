# Steady state support functions.
# Julia port of Python SSJ blocks/support/steady_state.py

using Roots: find_zero, Brent, Bisection, Newton
using NLsolve: nlsolve

# Instantiate mutable kwargs from Nothing defaults.
function instantiate_steady_state_mutable_kwargs(dissolve, block_kwargs, solver_kwargs, constrained_kwargs)
    isnothing(dissolve)           && (dissolve           = [])
    isnothing(block_kwargs)       && (block_kwargs       = Dict{String,Any}())
    isnothing(solver_kwargs)      && (solver_kwargs      = Dict())
    isnothing(constrained_kwargs) && (constrained_kwargs = Dict())
    return dissolve, block_kwargs, solver_kwargs, constrained_kwargs
end

# Infer default solver from unknowns structure (brentq for 1D with bounds, broyden_custom otherwise).
function provide_solver_default(unknowns)
    if length(unknowns) == 1
        bounds = first(values(unknowns))
        if !(bounds isa Tuple) || bounds[1] > bounds[2]
            error("Unable to find a compatible 1D solver. Provide bounds: unknowns = Dict(\"a\" => (0, 1))")
        end
        return "brentq"
    elseif length(unknowns) > 1
        init_vals = collect(values(unknowns))
        if !all(v isa Real for v in init_vals)
            error("For multi-dim solver provide scalar initial values: unknowns = Dict(\"a\"=>1., \"b\"=>2.)")
        end
        return "broyden_custom"
    else
        error("`unknowns` is empty!")
    end
end

# Check if residual is within tolerance.
function run_consistency_check(cresid, ctol=1e-9, fragile=false)
    if cresid > ctol
        msg = "Target residual $cresid > ctol $ctol. Check HelperBlocks or adjust ctol."
        fragile ? error(msg) : @warn msg
    end
end

# Compute target residuals (handles dict targets, string cross-refs, univariate shortcut).
function compute_target_values(targets, potential_args)
    target_values = zeros(length(targets))
    targets_vec   = targets isa Dict ? keys(targets) : targets
    for (i, t) in enumerate(targets_vec)
        v = targets isa Dict ? targets[t] : 0
        if v isa AbstractString
            target_values[i] = potential_args[t] - potential_args[v]
        else
            target_values[i] = potential_args[t] - v
        end
    end
    return length(targets) == 1 ? target_values[1] : target_values
end

# Compare two steady-state dicts element-wise.
function compare_steady_states(ss_ref, ss_comp; tol=1e-8, name_map=Dict(),
                                internal=true, check_same_keys=true, verbose=false)
    valid = true
    ds_to_check = [(ss_ref, ss_comp, "toplevel")]
    if internal && hasproperty(ss_ref, :internal) && hasproperty(ss_comp, :internal)
        for i in eachindex(ss_ref.internal)
            push!(ds_to_check, (ss_ref.internal[i], ss_comp.internal[i], string(i)*"_internal"))
        end
    end
    for (d_ref, d_comp, level) in ds_to_check
        for key_ref in keys(d_ref)
            key_comp = haskey(d_comp, key_ref) ? key_ref :
                       haskey(name_map, key_ref) ? name_map[key_ref] : continue
            resid = d_ref[key_ref] isa Number ?
                    abs(d_ref[key_ref] - d_comp[key_comp]) :
                    maximum(abs, d_ref[key_ref] .- d_comp[key_comp])
            verbose ? println("$key_ref resid: $resid") : (isapprox(resid, 0; atol=tol) || (valid = false))
        end
        if check_same_keys
            diff = symdiff(Set(keys(d_ref)) - Set(keys(name_map)),
                           Set(keys(d_comp)) - Set(values(name_map)))
            if !isempty(diff)
                verbose && println("Level '$level': keys in only one dict: $diff")
                valid = false
            end
        end
    end
    return valid
end

isscalar(x) = x isa Union{Number, AbstractString, Char, Bool}

# Extract bracket or x0 for 1D solvers.
function extract_univariate_initial_values_or_bounds(unknowns)
    val = first(values(unknowns))
    return isscalar(val) ? Dict("x0" => val) : Dict("bracket" => (val[1], val[2]))
end

# Extract initial values and bounds for multi-dim solvers.
function extract_multivariate_initial_values_and_bounds(unknowns; fragile=false)
    initial_values = Float64[]
    multi_bounds   = Dict{String,Tuple{Float64,Float64}}()
    for (k, v) in pairs(unknowns)
        if isscalar(v)
            push!(initial_values, Float64(v))
        elseif length(v) == 2
            fragile && error("Length-2 tuple for '$k' is ambiguous. Use length-3 (lb, x0, ub).")
            @warn "Interpreting length-2 tuple for '$k' as bounds; using midpoint as x0."
            push!(initial_values, (v[1] + v[2]) / 2.0)
        elseif length(v) == 3
            lb, iv, ub = v
            @assert lb < iv < ub "Bounds must satisfy lb < x0 < ub for '$k'"
            push!(initial_values, Float64(iv))
            multi_bounds[string(k)] = (Float64(lb), Float64(ub))
        else
            error("Invalid length $(length(v)) for unknown '$k'. Use scalar, (lb,ub), or (lb,x0,ub).")
        end
    end
    return initial_values, multi_bounds
end

# Bounds for multivariate solvers.
#
# Outside the box the residual is evaluated at the nearest point just inside it and
# inflated in proportion to the distance, which steers the solver back. The inflation
# is proportional to the residual itself, so a solver CAN stop at a point outside the
# box whose image inside the box is a root. `censor` maps such a point back to the
# root that was actually found; solve_for_unknowns applies it to the solution, so the
# returned unknowns always respect the bounds. (Previously the raw out-of-bounds point
# was returned and the steady state was re-evaluated there.)
function _censor(x, lbs, ubs, boundary_epsilon)
    x_cens = collect(Float64, x)
    for i in eachindex(x_cens)
        if x_cens[i] < lbs[i]; x_cens[i] = lbs[i] + boundary_epsilon
        elseif x_cens[i] > ubs[i]; x_cens[i] = ubs[i] - boundary_epsilon; end
    end
    return x_cens
end

function residual_with_linear_continuation(residual, bounds, names=collect(keys(bounds)); verbose=false,
                                            boundary_epsilon=1e-4, penalty_scale=1e1)
    # bounds are matched to unknowns BY NAME: unknowns without bounds are unconstrained
    lbs = [haskey(bounds, string(k)) ? bounds[string(k)][1] : -Inf for k in names]
    ubs = [haskey(bounds, string(k)) ? bounds[string(k)][2] :  Inf for k in names]
    cache = Ref{Any}(nothing)
    censor(x) = _censor(x, lbs, ubs, boundary_epsilon)

    function constr_residual(x)
        x_cens = censor(x)
        r = try residual(x_cens) catch e; (e isa InterruptException && rethrow()); nothing end
        if isnothing(r) || any(isnan.(r))
            r = isnothing(cache[]) ? ones(length(x)) : cache[] .* penalty_scale
        else
            cache[] = r
        end
        return r .+ (x .- x_cens) .* penalty_scale .* r
    end
    return constr_residual, censor
end

# Wrap residual with bound constraints. Returns the wrapped residual and the map
# from solver iterates to in-bounds points.
function constrained_multivariate_residual(residual, bounds, names=collect(keys(bounds)); method="linear_continuation", kwargs...)
    method == "linear_continuation" ||
        error("Constrained method '$method' not implemented.")
    return residual_with_linear_continuation(residual, bounds, names; kwargs...)
end

function _maybe_constrain(residual_f, bounds, unknowns, constrained_method, constrained_kwargs)
    isempty(bounds) && return residual_f, identity
    return constrained_multivariate_residual(residual_f, bounds, [string(k) for k in keys(unknowns)];
                                             method=constrained_method, constrained_kwargs...)
end

function _finalize_constrained(sol, censor, verbose)
    x = collect(Float64, sol)
    xc = censor === identity ? x : censor(x)
    if xc != x
        @warn "Steady-state solver stopped outside the bounds; returning the in-bounds point it actually evaluated: $(xc)"
    end
    return xc
end

# Univariate solvers mapping Python names → Roots.jl methods
const _UNI_SOLVERS = ["bisect", "brentq", "brenth", "ridder", "toms748", "newton", "secant", "halley"]
const _MULTI_SOLVERS = ["hybr", "lm", "broyden1", "broyden2", "anderson",
                        "linearmixing", "diagbroyden", "excitingmixing", "krylov", "df-sane"]

# Dispatch to correct solver. Matches Python solve_for_unknowns exactly.
function solve_for_unknowns(residual, unknowns, solver, solver_kwargs;
                             residual_kwargs=Dict(), constrained_method="linear_continuation",
                             constrained_kwargs=Dict(), tol=2e-12, verbose=false)

    isnothing(solver) && error("Must provide a solver: brentq, broyden_custom, newton_custom, hybr, solved")

    # Build residual with any fixed kwargs
    residual_f = isempty(residual_kwargs) ? residual :
                 (x -> residual(x; residual_kwargs...))

    if solver ∈ _UNI_SOLVERS
        # --- Univariate solver via Roots.jl ---
        iv_or_bounds = extract_univariate_initial_values_or_bounds(unknowns)
        if haskey(iv_or_bounds, "bracket")
            a, b = iv_or_bounds["bracket"]
            root = find_zero(residual_f, (a, b), Brent(); atol=tol)
        else
            root = find_zero(residual_f, iv_or_bounds["x0"], Newton(); atol=tol)
        end
        unknown_solutions = [root]

    elseif solver ∈ _MULTI_SOLVERS || solver == "hybr"
        # --- Multi-dim via NLsolve.jl ---
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        res_wrapped, censor = _maybe_constrain(residual_f, bounds, unknowns, constrained_method, constrained_kwargs)
        # NLsolve expects in-place F! form
        function f!(F, x); F .= res_wrapped(x); end
        nlsol = nlsolve(f!, initial_values; ftol=tol)
        nlsol.f_converged || @warn "NLsolve did not converge (solver=$solver)"
        unknown_solutions = _finalize_constrained(nlsol.zero, censor, verbose)

    elseif solver == "broyden_custom" || solver == "broyden"
        # --- Custom Broyden ---
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        res_wrapped, censor = _maybe_constrain(residual_f, bounds, unknowns, constrained_method, constrained_kwargs)
        sol, _ = broyden_solver(res_wrapped, initial_values; tol=tol, verbose=verbose)
        unknown_solutions = _finalize_constrained(sol, censor, verbose)

    elseif solver == "newton_custom" || solver == "newton"
        # --- Custom Newton ---
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        res_wrapped, censor = _maybe_constrain(residual_f, bounds, unknowns, constrained_method, constrained_kwargs)
        sol, _ = newton_solver(res_wrapped, initial_values; tol=tol, verbose=verbose)
        unknown_solutions = _finalize_constrained(sol, censor, verbose)

    elseif solver == "solved"
        # --- Pre-solved: just evaluate once to populate ss ---
        residual_f(collect(values(unknowns)))
        unknown_solutions = collect(values(unknowns))

    else
        error("steady_state is not yet compatible with solver '$solver'.")
    end

    return Dict(smart_zip(collect(keys(unknowns)), unknown_solutions))
end

# Helper: also needed from block.jl
