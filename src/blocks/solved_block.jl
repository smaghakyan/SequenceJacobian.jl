# A SolvedBlock wraps an inner block, solving for unknowns to hit targets numerically.

mutable struct SolvedBlock <: ParentBlock
    M
    steady_state_options
    solve_steady_state_options
    impulse_nonlinear_options
    solve_impulse_nonlinear_options
    impulse_linear_options
    jacobian_options
    partial_jacobians_options
    # Inner block being wrapped
    block
    name
    unknowns
    targets
    solver
    solver_kwargs
    # inputs and outputs after unknowns are endogenized
    inputs
    outputs
    kids
    descendants

    function SolvedBlock(block, name, unknowns, targets; solver=nothing, solver_kwargs=Dict())
        _M = Bijection(Dict())
        _steady_state_options = Dict()
        _solve_steady_state_options = Dict{String, Any}(
            "solver" => "",
            "solver_kwargs" => Dict(),
            "ttol" => 1e-12,
            "ctol" => 1e-9,
            "verbose" => false,
            "constrained_method" => "linear_continuation",
            "constrained_kwargs" => Dict())
        _impulse_nonlinear_options = Dict()
        _solve_impulse_nonlinear_options = Dict{String, Any}(
            "tol" => 1e-8,
            "maxit" => 30,
            "verbose" => true,
            "backtrack" => true,
            "max_backtracks" => 25,
            "warm_start" => false)
        _impulse_linear_options = Dict()
        _jacobian_options = Dict()
        _partial_jacobians_options = Dict()

        _block = block
        _name = (length(name) > 0 && name[1] == '_') ? name[2:end] : name
        _unknowns = unknowns
        _targets = targets
        _solver = solver
        _solver_kwargs = solver_kwargs
        _name, _kids, _descendants = construct_parent([block]; name=_name)

        if length(unknowns) ≠ length(targets)
            error("Unknowns and targets different sizes in SolvedBlock")
        elseif Set(keys(unknowns)) ⊈ block.inputs
            error("Unknowns has element not in inputs in SolvedBlock")
        elseif Set(targets) ⊈ block.outputs
            error("Target has element not in outputs in SolvedBlock")
        end

        _outputs = block.outputs ∪ Set(keys(_unknowns))
        _inputs  = setdiff(block.inputs, Set(keys(_unknowns)))

        return new(_M,
                   _steady_state_options,
                   _solve_steady_state_options,
                   _impulse_nonlinear_options,
                   _solve_impulse_nonlinear_options,
                   _impulse_linear_options,
                   _jacobian_options,
                   _partial_jacobians_options,
                   _block,
                   _name,
                   _unknowns,
                   _targets,
                   _solver,
                   _solver_kwargs,
                   _inputs,
                   _outputs,
                   _kids,
                   _descendants)
    end
end

function solved(f, unknowns, targets; solver=nothing, solver_kwargs=Dict(), name="")
    inner = rename(simple(f); name=String(nameof(f)) * "_inner")
    return SolvedBlock(inner, String(nameof(f)), unknowns, targets;
                       solver=solver, solver_kwargs=solver_kwargs)
end

Base.show(io::IO, sb::SolvedBlock) = print(io, "<SolvedBlock '$(sb.name)'>")

function _steady_state(sb::SolvedBlock, calibration; dissolve=[], options=Dict(), kwargs...)
    # Solver settings given to an enclosing solve must not replace this block's own.
    kw = Dict{Symbol,Any}(k => v for (k, v) in kwargs if k ∉ (:solver, :solver_kwargs))
    if sb.name ∈ dissolve
        unknowns = Dict(k => v for (k, v) in calibration if k ∈ keys(sb.unknowns))
        return solve_steady_state(sb.block, calibration, unknowns, sb.targets;
                                  options=options, kw..., solver="solved")
    else
        unknowns = sb.unknowns
        slv = isnothing(sb.solver) ? "" : sb.solver
        skw = isnothing(sb.solver_kwargs) ? Dict() : sb.solver_kwargs
        return solve_steady_state(sb.block, calibration, unknowns, sb.targets;
                                  options=options, kw..., solver=slv, solver_kwargs=skw)
    end
end

function _impulse_nonlinear(sb::SolvedBlock, ss, inputs; outputs, internals=Dict{String,Any}(),
                             Js=Dict(), options=Dict(), ss_initial=nothing, kwargs...)
    # the inner solve is silent unless asked otherwise (it runs once per outer iteration)
    kw = Dict{Symbol,Any}(kwargs)
    haskey(kw, :verbose) || (kw[:verbose] = false)
    return _dyn(solve_impulse_nonlinear,
        sb.block, ss, OrderedSet{String}(keys(sb.unknowns)), OrderedSet{String}(sb.targets), inputs;
        outputs=outputs, internals=internals, Js=Js, options=options,
        H_U_factored=_get_H_U_factored(sb, Js), ss_initial=ss_initial, kw...)
end

# Python: return self.block.solve_impulse_linear(ss, unknowns, targets, inputs, outputs, Js, options, H_U_factored)
function _impulse_linear(sb::SolvedBlock, ss, inputs, outputs, Js, options)
    return _dyn(solve_impulse_linear, sb.block, ss,
                                OrderedSet{String}(keys(sb.unknowns)),
                                OrderedSet{String}(sb.targets),
                                inputs, outputs;
                                Js=Js, options=options,
                                H_U_factored=_get_H_U_factored(sb, Js))
end

function _jacobian(sb::SolvedBlock, ss, inputs, outputs; T, Js=Dict(), options=Dict())
    return solve_jacobian(
        sb.block, ss, OrderedSet{String}(keys(sb.unknowns)), OrderedSet{String}(sb.targets),
        inputs, outputs; T=T, Js=Js, options=options,
        H_U_factored=_get_H_U_factored(sb, Js))[outputs]
end

function _partial_jacobians(sb::SolvedBlock, ss, inputs, outputs; T, Js=Dict(), options=Dict())
    inner_Js = partial_jacobians(
        sb.block, ss,
        OrderedSet{String}(keys(sb.unknowns)) ∪ inputs,
        OrderedSet{String}(sb.targets) ∪ setdiff(outputs, keys(sb.unknowns));
        T=T, Js=Js, options=options)
    H_U = jacobian(sb.block, ss, OrderedSet{String}(keys(sb.unknowns)), OrderedSet{String}(sb.targets);
                   T=T, Js=inner_Js, options=options)
    H_U_factored = FactoredJacobianDict(H_U; T=T)
    return merge(inner_Js, Dict(sb.name => H_U_factored))
end

function _get_H_U_factored(sb::SolvedBlock, Js)
    if sb.name ∈ keys(Js) && isa(Js[sb.name], FactoredJacobianDict)
        return Js[sb.name]
    else
        return nothing
    end
end
