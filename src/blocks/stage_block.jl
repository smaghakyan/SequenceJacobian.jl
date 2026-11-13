# stage_block.jl: StageBlock, a heterogeneous-agent block whose period is a sequence
# of stages (exogenous shocks, discrete choices, continuous choices).
# Julia version of Python SSJ blocks/stage_block.py.
#
#   hh = StageBlock([ExogenousMaker("Pi", 1; name="shocks"),
#                    LogitChoice("V", ["Va"], 1, "taste_scale"; f=flow_u, name="choice"),
#                    Continuous1D(["V", "Va"], "a", consav; name="consav", hetoutputs=[...])];
#                   name="hh", backward_init=init, hetinputs=[...])
#
# Stages are listed in the order in which they happen within a period. Each stage
# reports its own outcomes; an outcome `x` of any stage is aggregated with the
# distribution at the beginning of that stage and becomes the block output `X`.
#
# Steady-state internals: ss.internals[name][stage name] holds the end-of-stage
# backward objects, the individual outcomes, "law_of_motion" and "D" (the
# distribution at the BEGINNING of the stage).

using OrderedCollections: OrderedSet
using LinearAlgebra: dot

mutable struct StageBlock <: Block
    M
    steady_state_options
    solve_steady_state_options
    impulse_nonlinear_options
    solve_impulse_nonlinear_options
    impulse_linear_options
    jacobian_options
    partial_jacobians_options

    stages::Vector{Stage}
    name::String
    inputs
    outputs
    M_outputs
    internals
    original_inputs
    hetinputs
    backward_init
    cache                # last steady-state solution, used when option "warm_start" is true
end

function StageBlock(stages; backward_init=nothing, hetinputs=nothing, name=nothing)
    stages = make_all_into_stages(stages)
    n = length(stages)

    inputs  = OrderedSet{String}()
    outputs = OrderedSet{String}()
    for (i, stage) in enumerate(stages)
        # external inputs: whatever a stage does not take from the next stage
        union!(inputs, setdiff(stage.inputs, stages[mod1(i+1, n)].backward_outputs))
        union!(outputs, stage.report)
    end
    _stage_block_checks(stages, inputs, outputs)

    nm = isnothing(name) ? stages[1].name * "_to_" * stages[end].name : string(name)
    hi = isnothing(hetinputs) ? nothing :
         (hetinputs isa AbstractCombinedExtendedFunction ? hetinputs : CombinedExtendedFunction(collect(make_tuple(hetinputs))))
    bi = isnothing(backward_init) ? nothing : _stage_fun(backward_init)

    sb = StageBlock(Bijection(Dict()),
        Dict{String,Any}("backward_tol" => 1e-9, "backward_maxit" => 5000,
                         "forward_tol" => 1e-10, "forward_maxit" => 100_000, "warm_start" => false),
        Dict{String,Any}("solver" => "", "solver_kwargs" => Dict(), "ttol" => 1e-12, "ctol" => 1e-9,
                         "verbose" => false, "constrained_method" => "linear_continuation",
                         "constrained_kwargs" => Dict()),
        Dict{String,Any}(),
        Dict{String,Any}("tol" => 1e-8, "maxit" => 30, "verbose" => true, "backtrack" => true,
                         "max_backtracks" => 25, "warm_start" => false),
        Dict{String,Any}("h" => 1e-4, "twosided" => false, "h_hetoutputs" => 1e-5),
        Dict{String,Any}("h" => 1e-4, "twosided" => false, "h_hetoutputs" => 1e-5),
        Dict{String,Any}(),
        stages, nm, inputs,
        OrderedSet{String}(uppercase(o) for o in outputs),
        Bijection(Dict(o => uppercase(o) for o in outputs)),
        OrderedSet{String}(), copy(inputs), nothing, bi, Dict{String,Any}())
    return _process_stage_hetinputs!(sb, hi)
end

function _stage_block_checks(stages, inputs, outputs)
    isempty(intersect(inputs, outputs)) || error("StageBlock: inputs and outputs overlap: $(collect(intersect(inputs, outputs)))")
    backward_all = OrderedSet{String}()
    for stage in stages; union!(backward_all, stage.backward_outputs); end
    isempty(intersect(inputs, backward_all)) || error("StageBlock: a stage takes a backward variable from a stage that is not its successor: $(collect(intersect(inputs, backward_all)))")
    isempty(intersect(outputs, backward_all)) || error("StageBlock: outputs and backward variables overlap: $(collect(intersect(outputs, backward_all)))")
    names = [stage.name for stage in stages]
    length(unique(names)) == length(names) || error("StageBlock: stage names must be unique, got $names")
    seen = OrderedSet{String}()
    for stage in stages
        stage.name in ("D", "law_of_motion") && error("Stage name '$(stage.name)' is reserved")
        for o in stage.report
            o in ("d", "law_of_motion") && error("Stage '$(stage.name)' may not report '$o'")
            isuppercase(o[1]) && error("Stage '$(stage.name)' reports '$o': individual outcomes must not start with an upper-case letter")
            o in seen && error("Outcome '$o' is reported by more than one stage")
            push!(seen, o)
        end
    end
end

function _process_stage_hetinputs!(sb::StageBlock, hetinputs)
    inp = copy(sb.original_inputs)
    if !isnothing(hetinputs)
        union!(inp, hetinputs.inputs)
        inp = OrderedSet{String}(setdiff(inp, hetinputs.outputs))
    end
    sb.inputs = inp
    sb.hetinputs = hetinputs
    return sb
end

function add_hetinputs(sb::StageBlock, functions)
    new = deepcopy(sb)
    hi = isnothing(sb.hetinputs) ? CombinedExtendedFunction(collect(make_tuple(functions))) : add(sb.hetinputs, functions)
    return _process_stage_hetinputs!(new, hi)
end

Base.show(io::IO, sb::StageBlock) = print(io, "<StageBlock '$(sb.name)' with stages $([s.name for s in sb.stages])>")

# ---- helpers -------------------------------------------------------------------
function _stage_extract_ss(sb::StageBlock, ss)
    if ss isa SteadyStateDict
        d = Dict{String,Any}(ss.toplevel)
        haskey(ss.internals, sb.name) && merge!(d, ss.internals[sb.name])
        return d
    end
    return Dict{String,Any}(ss)
end

function _stage_hetinputs(sb::StageBlock, d)
    isnothing(sb.hetinputs) && return Dict{String,Any}()
    res = sb.hetinputs(d)
    return Dict{String,Any}(k => res[k] for k in sb.hetinputs.outputs)
end

_tslice(A, t) = selectdim(A, 1, t)

# ---- steady state --------------------------------------------------------------
function _steady_state(sb::StageBlock, calibration; backward_tol=1e-9, backward_maxit=5000,
                       forward_tol=1e-10, forward_maxit=100_000, warm_start=false)
    ss = _stage_extract_ss(sb, calibration)
    hetin = _stage_hetinputs(sb, ss)
    merge!(ss, hetin)

    first_back = sb.stages[1].backward_outputs
    backward0 = nothing
    if warm_start && haskey(sb.cache, "backward") &&
            all(haskey(sb.cache["backward"], k) for k in first_back)
        backward0 = sb.cache["backward"]
    elseif all(haskey(ss, k) for k in first_back)
        backward0 = Dict{String,Any}(k => ss[k] for k in first_back)
    else
        isnothing(sb.backward_init) && error("StageBlock '$(sb.name)': no starting values for $(collect(first_back)); supply backward_init")
        init = sb.backward_init(ss)
        backward0 = Dict{String,Any}(k => init[k] for k in first_back)
    end

    backward, report, lom, backward_first = _stage_backward_ss(sb, backward0, ss, backward_tol, backward_maxit)

    Dinit = nothing
    if warm_start && haskey(sb.cache, "D") && size(sb.cache["D"]) == size(first(values(backward[end])))
        Dinit = sb.cache["D"]
    elseif haskey(ss, sb.stages[1].name) && ss[sb.stages[1].name] isa AbstractDict && haskey(ss[sb.stages[1].name], "D")
        Dinit = ss[sb.stages[1].name]["D"]
    else
        example = first(values(backward[end]))
        Dinit = fill(1.0 / length(example), size(example))
    end
    D = _stage_forward_ss(sb, Dinit, lom, forward_tol, forward_maxit)

    sb.cache["backward"] = backward_first
    sb.cache["D"] = D[1]

    aggregates = Dict{String,Any}()
    internals  = Dict{String,Any}(hetin)
    for (i, stage) in enumerate(sb.stages)
        for k in stage.report
            aggregates[uppercase(k)] = dot(D[i], report[i][k])
        end
        internals[stage.name] = merge(Dict{String,Any}(backward[i]), report[i],
                                      Dict{String,Any}("law_of_motion" => lom[i], "D" => D[i]))
    end
    for k in sb.inputs
        haskey(ss, k) && (aggregates[k] = ss[k])
    end
    return SteadyStateDict(aggregates; internals=Dict{String,Any}(sb.name => internals))
end

# one pass backward through all stages (no reports)
function _stage_backward_step(sb::StageBlock, backward, inputs)
    for stage in reverse(sb.stages)
        backward, _ = backward_step_separate(stage, merge(inputs, backward))
    end
    return backward
end

# one pass backward, keeping for every stage (in chronological order) the backward
# INPUTS, the reports and the law of motion; fourth value: backward output of stage 1
function _stage_backward_step_full(sb::StageBlock, backward, inputs)
    backward_all = Any[backward]; report_all = Any[]; lom_all = Any[]
    for stage in reverse(sb.stages)
        (backward, report), lom = backward_step_separate(stage, merge(inputs, backward); lawofmotion=true, hetoutputs=true)
        push!(backward_all, backward); push!(report_all, report); push!(lom_all, lom)
    end
    return reverse(backward_all)[2:end], reverse(report_all), reverse(lom_all), backward_all[end]
end

function _stage_backward_ss(sb::StageBlock, backward, ss, tol, maxit)
    converged = false
    for it in 1:maxit
        backward_new = _stage_backward_step(sb, backward, ss)
        if it % 10 == 1 && all(within_tolerance(backward_new[k], backward[k], tol) for k in keys(backward))
            backward = backward_new; converged = true; break
        end
        backward = backward_new
    end
    converged || @warn "StageBlock '$(sb.name)': no convergence after $(maxit) backward iterations."
    return _stage_backward_step_full(sb, backward, ss)
end

function _stage_forward_step(D, loms)
    for lom in loms
        D = forward(lom, D)
    end
    return D
end

function _stage_forward_step_full(D, loms)
    Ds = Any[D]
    for lom in loms
        push!(Ds, forward(lom, Ds[end]))
    end
    return Ds[1:end-1], Ds[end]
end

function _stage_forward_ss(sb::StageBlock, D, loms, tol, maxit)
    converged = false
    for it in 1:maxit
        D_new = _stage_forward_step(D, loms)
        if it % 10 == 1 && within_tolerance(D, D_new, tol)
            D = D_new; converged = true; break
        end
        D = D_new
    end
    converged || @warn "StageBlock '$(sb.name)': no convergence after $(maxit) forward iterations."
    return _stage_forward_step_full(D, loms)[1]
end

# ---- nonlinear transition --------------------------------------------------------
# Two sweeps. Backward, from date T to date 1: only the value functions that enter each
# date are kept. Forward, from date 1 to date T: policies and outcomes of each date are
# computed again from those value functions, aggregated with the distribution, and the
# distribution is moved on. Memory is T x (state space) x (number of value functions);
# keeping the outcomes and laws of motion of every date instead takes about ten times
# as much for a block with many outcomes.
function _impulse_nonlinear(sb::StageBlock, ssin, inputs; outputs, internals=String[], ss_initial=nothing)
    ss = _stage_extract_ss(sb, ssin)
    first_name = sb.stages[1].name
    haskey(ss, first_name) || error("Steady state passed to '$(sb.name)' has no internals: compute it with steady_state first.")
    Dbeg = ss[first_name]["D"]
    if !isnothing(ss_initial)
        init = ss_initial isa SteadyStateDict ? ss_initial.internals[sb.name] : ss_initial
        Dbeg = init[first_name]["D"]
    end
    T = inputs.T
    out_names = OrderedSet{String}(string(o) for o in outputs)
    wanted = [[o for o in stage.report if uppercase(o) in out_names] for stage in sb.stages]

    indict = copy(ss)
    function set_date!(t)
        for (k, v) in inputs.toplevel
            indict[string(k)] = ss[string(k)] .+ v[t]
        end
        merge!(indict, _stage_hetinputs(sb, indict))
        return indict
    end

    # backward sweep
    cont = Vector{Any}(undef, T)                 # value functions that enter date t (they come from date t + 1)
    backward = Dict{String,Any}(k => ss[sb.stages[end].name][k] for k in sb.stages[1].backward_outputs)
    for t in T:-1:1
        cont[t] = backward
        backward = _stage_backward_step(sb, backward, set_date!(t))
    end

    # forward sweep
    agg = Dict{String,Any}()
    for (j, stage) in enumerate(sb.stages), o in wanted[j]
        agg[uppercase(o)] = zeros(T)
    end
    for t in 1:T
        _, report, lom, _ = _stage_backward_step_full(sb, cont[t], set_date!(t))
        Ds, Dbeg = _stage_forward_step_full(Dbeg, lom)
        for (j, stage) in enumerate(sb.stages), o in wanted[j]
            agg[uppercase(o)][t] = dot(Ds[j], report[j][o])
        end
    end
    for (j, stage) in enumerate(sb.stages), o in wanted[j]
        agg[uppercase(o)] .-= dot(ss[stage.name]["D"], ss[stage.name][o])
    end
    return ImpulseDict(agg; T=T)
end

function _impulse_linear(sb::StageBlock, ss, inputs, outputs, Js; h=1e-4, twosided=false, h_hetoutputs=1e-5)
    T = inputs.T
    in_names = OrderedSet{String}(string(k) for k in keys(inputs.toplevel))
    J = _jacobian(sb, ss, in_names, outputs; T=T, h=h, twosided=twosided, h_hetoutputs=h_hetoutputs)
    return apply(J, inputs)[OrderedSet{String}(string(o) for o in J.outputs)]
end

# ---- Jacobian: fake news algorithm -------------------------------------------------
# Step sizes follow the Python toolbox: 1e-4 for the stage functions and hetinputs,
# 1e-5 for hetoutputs; one-sided differences unless `twosided=true`.
function _jacobian(sb::StageBlock, ss, inputs, outputs; T, h=1e-4, twosided=false, h_hetoutputs=1e-5)
    ss = _stage_extract_ss(sb, ss)
    outputs = OrderedSet{String}(string(o) for o in inv(sb.M_outputs) * OrderedSet{String}(string(o) for o in outputs))
    inputs  = [string(i) for i in inputs]
    diff_hetinput = isnothing(sb.hetinputs) ? nothing : differentiable(sb.hetinputs, ss; h=h, twosided=twosided)
    backward_data, forward_data, expectations_data = _stage_prelim(sb, ss, h, twosided, h_hetoutputs)

    # Step 1 for every input: one array of T x (state space) per input is kept. The
    # expectation vectors (step 2) are then computed for one output at a time, so that
    # memory grows with the number of inputs and not with the number of outputs
    curlyY = Dict{String,Any}(); curlyD = Dict{String,Any}()
    for i in inputs
        Ys, Ds = _stage_backward_fakenews(sb, i, outputs, T, backward_data, forward_data, diff_hetinput)
        curlyY[i] = Ys
        curlyD[i] = reshape(Ds, T, :)
    end

    J = Dict{String,Dict{String,Any}}(uppercase(o) => Dict{String,Any}() for o in outputs)
    for o in outputs                                                             # steps 2-4
        E = reshape(_stage_expectation_vectors(o, T-1, expectations_data), T-1, :)
        for i in inputs
            F = Array{Float64}(undef, T, T)
            F[1, :] = get(curlyY[i], o, zeros(T))
            F[2:end, :] = E * transpose(curlyD[i])
            J[uppercase(o)][i] = J_from_F(F)
        end
    end
    return JacobianDict(J; name=sb.name, T=T)
end

function _stage_prelim(sb::StageBlock, ss, h, twosided, h_hetoutputs)
    backward_data = Any[]; forward_data = Any[]; expectations_data = Any[]
    for stage in reverse(sb.stages)
        sdict = ss[stage.name]
        potential = merge(ss, sdict)                       # stage internals take precedence
        input  = Dict{String,Any}(k => potential[k] for k in stage.inputs)
        report = Dict{String,Any}(k => sdict[k] for k in stage.report)
        D   = sdict["D"]
        lom = sdict["law_of_motion"]
        pre = precompute(stage, input, lom; h=h, twosided=twosided)
        ho = nothing
        if !isnothing(stage.hetoutputs)
            ho_in = Dict{String,Any}(k => potential[k] for k in stage.hetoutputs.inputs)
            ho = differentiable(stage.hetoutputs, ho_in; h=h_hetoutputs, twosided=twosided)
        end
        push!(backward_data, (stage, input, D, lom, pre, ho))
        push!(forward_data, (stage, report, D, lom))
        push!(expectations_data, (report, lom))
    end
    return backward_data, reverse(forward_data), expectations_data
end

function _stage_backward_fakenews(sb, input_shocked, output_list, T, backward_data, forward_data, diff_hetinput)
    din = Dict{String,Any}(input_shocked => 1.0)
    if !isnothing(diff_hetinput) && input_shocked in diff_hetinput.inputs
        merge!(din, diff(diff_hetinput, Dict{String,Any}(input_shocked => 1.0)))
    end
    curlyV, curlyD, curlyY = _stage_backward_step_fakenews(din, output_list, backward_data, forward_data)

    curlyDs = Array{Float64}(undef, T, size(curlyD)...)
    curlyYs = Dict{String,Any}(k => zeros(T) for k in keys(curlyY))
    _tslice(curlyDs, 1) .= curlyD
    for k in keys(curlyY); curlyYs[k][1] = curlyY[k]; end
    for t in 2:T
        curlyV, curlyD, curlyY = _stage_backward_step_fakenews(curlyV, output_list, backward_data, forward_data)
        _tslice(curlyDs, t) .= curlyD
        for k in keys(curlyY)
            haskey(curlyYs, k) || (curlyYs[k] = zeros(T))
            curlyYs[k][t] = curlyY[k]
        end
    end
    return curlyYs, curlyDs
end

# Given shocks to this period's inputs: perturbation of the backward outputs of the
# first stage (curlyV), of the end-of-period distribution (curlyD) and of the
# aggregate outputs (curlyY).
function _stage_backward_step_fakenews(din, output_list, backward_data, forward_data)
    dback = Dict{String,Any}()
    dloms = Any[]
    curlyY = Dict{String,Float64}()

    # backward through the stages: shocks to laws of motion, and the part of curlyY
    # that does not come through the distribution
    for (stage, ssin, D, lom, pre, ho) in backward_data
        din_all = merge(din, dback)
        dout, dlom = backward_step_shock(stage, ssin, din_all, pre)
        push!(dloms, dlom)
        dback = Dict{String,Any}(k => dout[k] for k in stage.backward_outputs if haskey(dout, k))
        if !isnothing(ho)
            want = [k for k in ho.outputs if k in output_list]
            if !isempty(want)
                merge!(dout, diff(ho, merge(din_all, dout); outputs=want))
            end
        end
        for k in stage.report
            (k in output_list && haskey(dout, k)) && (curlyY[k] = dot(D, dout[k]))
        end
    end
    curlyV = dback

    # forward through the stages: perturbation of the distribution
    dD = nothing
    Dsize = size(forward_data[1][3])
    for ((stage, ssrep, D, lom), dlom) in zip(forward_data, reverse(dloms))
        if !isnothing(dD)
            for k in stage.report
                k in output_list && (curlyY[k] = get(curlyY, k, 0.0) + dot(dD, ssrep[k]))
            end
            dD = forward(lom, dD)
            isnothing(dlom) || (dD = dD .+ forward_shock(dlom, D))
        elseif !isnothing(dlom)
            dD = forward_shock(dlom, D)
        end
    end
    return curlyV, (isnothing(dD) ? zeros(Dsize) : dD), curlyY
end

# Expected value of output o, at the beginning of the first stage, 0..T-1 periods ahead
# (demeaned for numerical reasons; this does not affect products with curlyD).
function _stage_expectation_vectors(o, T, expectations_data)
    cur = nothing
    for (ssrep, lom) in expectations_data
        isnothing(cur) || (cur = expectation(lom, cur))
        haskey(ssrep, o) && (cur = ssrep[o])
    end
    isnothing(cur) && error("Output '$o' is not reported by any stage")
    curlyEs = Array{Float64}(undef, T, size(cur)...)
    _tslice(curlyEs, 1) .= demean(cur)
    for t in 2:T
        prev = collect(_tslice(curlyEs, t-1))
        for (_, lom) in expectations_data
            prev = expectation(lom, prev)
        end
        _tslice(curlyEs, t) .= demean(prev)
    end
    return curlyEs
end
