mutable struct HetBlock <: Block
    M
    steady_state_options
    solve_steady_state_options
    impulse_nonlinear_options
    solve_impulse_nonlinear_options
    impulse_linear_options
    jacobian_options
    partial_jacobian_options

    backward_fun
    name
    exogenous
    policy
    backward
    non_backward_outputs

    outputs
    M_outputs
    inputs
    internals

    #not needed yet
    #static_checks

    original_inputs
    original_outputs
    original_internals
    original_M_outputs

    hetinputs
    hetoutputs
   # process_hetinputs_hetoutputs

    backward_init

    # Constructor for HetBlock, hard coded static checks
    function HetBlock(backward_fun, exogenous, policy, backward, backward_init=nothing, hetinputs=nothing, hetoutputs = nothing)
    _M = Bijection(Dict())
    # Defaults follow the Python toolbox. They can be overridden per call
    # (`steady_state(hh, cal; backward_tol=1e-10)`), per model call
    # (`options=Dict("hh" => Dict("backward_tol" => 1e-10))`), or on the block
    # (`hh.steady_state_options["backward_tol"] = 1e-10`).
    _steady_state_options = Dict{String, Any}(
        "backward_tol" => 1e-8,
        "backward_maxit" => 10_000,
        "forward_tol" => 1e-10,
        "forward_maxit" => 100_000)
    _solve_steady_state_options = Dict{String, Any}(
        "solver" => "",
        "solver_kwargs" => Dict(),
        "ttol" => 1e-12,
        "ctol" => 1e-9,
        "verbose" => false,
        "constrained_method" => "linear_continuation",
        "constrained_kwargs" => Dict())
    _impulse_nonlinear_options = Dict{String, Any}("monotonic" => false)
    _solve_impulse_nonlinear_options = Dict{String, Any}(
        "tol" => 1e-8,
        "maxit" => 30,
        "verbose" => true,
        "backtrack" => true,
        "max_backtracks" => 25,
        "warm_start" => false)
    _impulse_linear_options = Dict{String, Any}("h" => 1e-4, "twosided" => false)
    _jacobian_options = Dict{String, Any}("h" => 1e-4, "twosided" => false)
    _partial_jacobians_options = Dict{String, Any}()

    _backward_fun = ExtendedFunction(backward_fun)
    _name = _backward_fun.name
    _exogenous = OrderedSet{String}(make_tuple(exogenous))
    _policy, _backward = (OrderedSet{String}(make_tuple(x)) for x ∈ (policy, backward))

    # Ensure policy and backward variables are always in outputs.
    # output_list may return empty (source not on disk) or partial results.
    # We always guarantee the minimum required outputs are present.
    guaranteed = OrderedSet{String}(_policy ∪ _backward)
    all_outputs = _backward_fun.outputs ∪ guaranteed
    if all_outputs != _backward_fun.outputs
        _backward_fun = ExtendedFunction(
            _backward_fun.f, _backward_fun.name, _backward_fun.inputs, all_outputs)
    end

    _non_backward_outputs = setdiff(_backward_fun.outputs, _backward)

    _outputs = OrderedSet{String}(uppercase(o) for o ∈ _non_backward_outputs)
    _M_outputs = Bijection(Dict(o => uppercase(o) for o ∈ _non_backward_outputs))
    _inputs = OrderedSet{String}(collect(setdiff(Set(_backward_fun.inputs), Set([k * "_p" for k in _backward]))))
    _inputs = _inputs ∪ _exogenous
    _internals = OrderedSet{String}(["D", "Dbeg"]) ∪ _exogenous ∪ _backward_fun.outputs

    _original_inputs = _inputs
    _original_outputs = _outputs
    _original_internals = _internals
    _original_M_outputs = _M_outputs

    #static_checks
  if length(_policy) > 2
        error("More than two endogenous policies in '$(_name)', not yet supported")
    end

    for pol ∈ _policy
        if !(pol ∈ _backward_fun.outputs)
            error("Policy '$(pol)' not included as output in '$(_name)'")
        end
        if isuppercase(pol[1])
            error("Policy '$(pol)' is uppercase in '$(_name)', which is not allowed")
        end
    end

    for back ∈ _backward
        if !(back * "_p" ∈ _backward_fun.inputs)
            error("Backward variable '$(back)'_p not included as argument in '$(_name)'")
        end
        if !(back ∈ _backward_fun.outputs)
            error("Backward variable '$(back)' not included as output in '$(_name)'")
        end
        if back ∈ ["d", "dbeg", "D", "Dbeg"]
            error("A backward variable is called D or Dbeg, which are reserved for the distribution")
        end
    end

    for out ∈ _non_backward_outputs
        if isuppercase(out[1])
            error("Output '$(out)' is uppercase in '$(_name)', which is not allowed")
        end
    end
# HetBlock can have heterogeneous inputs and outputs, so we have hetinput and hetoutput
    if !isnothing(hetinputs)
        hetinputs = CombinedExtendedFunction(hetinputs)
    end
    if !isnothing(hetoutputs)
        hetoutputs = CombinedExtendedFunction(hetoutputs)
    end

# Process_hetinputs_hetoutputs hard-coded
   if !isnothing(hetoutputs)
        _inputs = _inputs ∪ setdiff(hetoutputs.inputs, _backward_fun.outputs ∪ OrderedSet{String}(["D"]))
        _outputs = _outputs ∪ OrderedSet{String}(uppercase(o) for o ∈ hetoutputs.outputs)
        _M_outputs = Bijection(Dict(o => uppercase(o) for o ∈ hetoutputs.outputs)) * _original_M_outputs
        # the individual-level hetoutput arrays are internals (this line used to add
        # the block's *inputs* instead, so e.g. `chi` leaked into the top level)
        _internals = _internals ∪ hetoutputs.outputs
    end
    if !isnothing(hetinputs)
        _inputs = _inputs ∪ hetinputs.inputs
        _inputs = setdiff(_inputs, hetinputs.outputs)
        _internals = _internals ∪ hetinputs.outputs
    end
#end of process_hetinputs_hetoutputs

    if !isnothing(backward_init)
        backward_init = ExtendedFunction(backward_init)
    end
    _backward_init = backward_init

    new(_M,
        _steady_state_options,
        _solve_steady_state_options,
        _impulse_nonlinear_options,
        _solve_impulse_nonlinear_options,
        _impulse_linear_options,
        _jacobian_options,
        _partial_jacobians_options,
        _backward_fun,
        _name,
        _exogenous,
        _policy,
        _backward,
        _non_backward_outputs,
        _outputs,
        _M_outputs,
        _inputs,
        _internals,
        _original_inputs,
        _original_outputs,
        _original_internals,
        _original_M_outputs,
        hetinputs,
        hetoutputs,
        _backward_init
        )
    end
end

function het(backward_fun; exogenous, policy, backward, backward_init=nothing, hetinputs=nothing, hetoutputs=nothing)
    return HetBlock(backward_fun, exogenous, policy, backward, backward_init, hetinputs, hetoutputs)
end



function Base.show(io::IO, b::HetBlock)
    if !isnothing(b.hetinputs)
        if !isnothing(b.hetoutputs)
            print(io, "<HetBlock '$(b.name)' with hetinput '$(b.hetinputs.name)' and with hetoutput '$(b.hetoutputs.name)'>")
        else
            print(io, "<HetBlock '$(b.name)' with hetinput '$(b.hetinputs.name)'>")
        end
    else
        print(io, "<HetBlock '$(b.name)'>")
    end
    return nothing
end

# Computes steady state of HetBlock with forward, backward computations on ss
function _steady_state(b::HetBlock, calibration; backward_tol=1e-8, backward_maxit=10_000, forward_tol=1e-10, forward_maxit=100_000)
    ss = extract_ss_dict(b, calibration)
    ss = update_with_hetinputs(b, ss)
    ss = initialize_backward(b, ss)

    ss = backward_steady_state(b, ss; tol=backward_tol, maxit=backward_maxit)
    Dbeg, D = forward_steady_state(b, ss; tol=forward_tol, maxit=forward_maxit)
    ss = merge!(ss, Dict("Dbeg" => Dbeg, "D" => D))

    ss = update_with_hetoutputs(b, ss)

    toreturn = b.non_backward_outputs
    if !isnothing(b.hetoutputs)
        toreturn = toreturn ∪ b.hetoutputs.outputs
    end
    aggregates = Dict{String,Any}(uppercase(string(o)) => dot(D, ss[string(o)]) for o ∈ toreturn)
    merge!(ss, aggregates)

    return SteadyStateDict(
        Dict{String,Any}(k => ss[k] for k ∈ keys(ss) if k ∉ b.internals);
        internals=Dict{String,Any}(b.name => Dict{String,Any}(k => ss[k] for k ∈ keys(ss) if k ∈ b.internals)))
end


# Partial-equilibrium linear impulse: apply the fake-news Jacobian to the input paths.
function _impulse_linear(b::HetBlock, ss, inputs, outputs, Js; h=1e-4, twosided=false)
    T = inputs.T
    in_names = OrderedSet{String}(string(k) for k in keys(inputs.toplevel))
    J = _jacobian(b, ss, in_names, outputs; T=T, h=h, twosided=twosided)
    return apply(J, inputs)[OrderedSet{String}(string(o) for o in J.outputs)]
end

# Implements fake news algorithm from SSJ paper
function _jacobian(b::HetBlock, ss, inputs, outputs; T, h=1e-4, twosided=false)
    ss = extract_ss_dict(b, ss)
    outputs = inv(b.M_outputs) * outputs

    # Step 0: preliminary processing of ss
    exog = make_exog_law_of_motion(b, ss)
    endog = make_endog_law_of_motion(b, ss)
    differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs = jac_backward_prelim(b, ss, h, exog, twosided)
    law_of_motion = forward_shockable(CombinedTransition([exog, endog]), ss["Dbeg"]) # fix this; not sure how CombTrans works
    exog_by_output = Dict{String,Any}(string(k) => expectation_shockable(exog, ss[string(k)]) for k ∈ (outputs ∪ b.backward))

    # Step 1 of Fake News Algo
    curlyYs, curlyDs = Dict{String,Any}(), Dict{String,Any}()
    for i ∈ inputs
        i_str = string(i)
        curlyYs[i_str], curlyDs[i_str] = backward_fakenews(b, i_str, outputs, T, differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs, law_of_motion, exog_by_output)
    end

    # Step 2 of Fake News Algo
    curlyPs = Dict{String,Any}()
    for o in outputs
        o_str = string(o)
        curlyPs[o_str] = expectation_vectors(b, ss[o_str], T, law_of_motion)
    end



    # Steps 3-4 of Fake News Algo
    F = Dict{String, Dict{String,Any}}()
    J = Dict{String, Dict{String,Any}}()
    for o in outputs
        o_str = string(o)
        O_str = uppercase(o_str)
        for i in inputs
            i_str = string(i)
            if !haskey(F, O_str); F[O_str] = Dict{String,Any}(); end
            if !haskey(J, O_str); J[O_str] = Dict{String,Any}(); end
            F[O_str][i_str] = build_F(curlyYs[i_str][o_str], curlyDs[i_str], curlyPs[o_str])
            J[O_str][i_str] = J_from_F(F[O_str][i_str])[1:T, 1:T]  # trim to T×T
        end
    end

    return JacobianDict(J; name=b.name, T=T)
end


function backward_steady_state(b::HetBlock, ss; tol=1e-8, maxit=10_000)
    ss = copy(ss)
    exog = make_exog_law_of_motion(b, ss)

    old = Dict{String,Any}()
    for it ∈ 1:maxit
        for k ∈ b.backward
            ss[k * "_p"] = expectation(exog, ss[k])
            delete!(ss, k)
        end
        ss = merge(ss, b.backward_fun(ss))

        if (mod(it, 10) == 2) && !isempty(old) &&
                all(within_tolerance(ss[k], old[k], tol) for k ∈ b.policy)
            break
        end
        old = merge(old, Dict(k => copy(ss[k]) for k in b.policy))
        if it == maxit
            @warn "No convergence of policy functions after $(maxit) backward iterations!"
        end
    end

    for k ∈ b.backward
        delete!(ss, k * "_p")
    end
    return ss
end

function forward_steady_state(b::HetBlock, ss; tol=1e-10, maxit=100_000)
    # Forward iteration to get steady state distribution
    exog = make_exog_law_of_motion(b, ss)
    endog = make_endog_law_of_motion(b, ss)

    Dbeg_seed = get(ss, "Dbeg", nothing)
    pi_seeds = [get(ss, k * "_seed", nothing) for k ∈ b.exogenous]

    # obtain initial Distribution D
    if isnothing(Dbeg_seed)
        # stationary distribution of each exogenous var
        pis = [stationary(exog[i], pi_seed) for (i, pi_seed) ∈ enumerate(pi_seeds)]

        # uniform distribution over endogenous vars
        endog_uniform = [fill(1/length(ss[k * "_grid"]), length(ss[k * "_grid"])) for k ∈ b.policy]

        # initialize outer product as guess
        Dbeg = outer(vcat(pis, endog_uniform))
    else
        Dbeg = Dbeg_seed
    end

    # iterate until convergence, or maxit reached
    D = forward(exog, Dbeg)
    for it in 1:maxit
        Dbeg_new = forward(endog, D)
        D_new = forward(exog, Dbeg_new)

        # check convergence every 10 iterations for efficiency purposes
        if it % 10 == 0 && within_tolerance(Dbeg, Dbeg_new, tol) # from optimized routines util
            break
        end
        Dbeg = Dbeg_new
        D = D_new

        if it == maxit
            @warn "No convergence after $(maxit) forward iterations! Results may be inaccurate."
        end
    end

    # D is after exogenous shock, Dbeg before it
    return Dbeg, D
end

function backward_fakenews(b::HetBlock, input_shocked, output_list, T, differentiable_backward_fun, differentiable_hetinput, differentiable_hetoutput, law_of_motion, exog)
    # Part 1 of Fake News Algo: calculate curlyY and curlyD in response to fake news shock
    din_dict = Dict{String,Any}(string(input_shocked) => 1)
    if !isnothing(differentiable_hetinput) && (input_shocked ∈ differentiable_hetinput.inputs)
        din_dict = merge(din_dict, diff(differentiable_hetinput, Dict{String,Any}(string(input_shocked) => 1)))
    end

    curlyV, curlyD, curlyY = backward_step_fakenews(b, din_dict, output_list, differentiable_backward_fun, differentiable_hetoutput, law_of_motion, exog; maybe_exog_shock=true)

    # infer dimensions, initialize empty arrays, fill in comtemporaneous effect
    curlyDs = Array{Float64}(undef, (T, size(curlyD)...))
    curlyDs[1, [Colon() for i ∈ 2:ndims(curlyDs)]...] = curlyD

    curlyYs = Dict{String,Any}(string(k) => Array{Float64}(undef, T) for k ∈ keys(curlyY))
    for k ∈  keys(curlyY)
        curlyYs[k][1] = curlyY[k]
    end

    for t ∈ 2:T
        curlyV, curlyDs[t, [Colon() for i ∈ 2:ndims(curlyDs)]...], curlyY = backward_step_fakenews(b, Dict{String,Any}(string(k) * "_p" => v for (k, v) ∈ pairs(curlyV)), output_list, differentiable_backward_fun, differentiable_hetoutput, law_of_motion, exog)
        for k ∈  keys(curlyY)
            curlyYs[k][t] = curlyY[k]
        end
    end
    return curlyYs, curlyDs
end

function expectation_vectors(block::HetBlock, o_ss, T::Int, law_of_motion::Transition)
    # Part 2 of the fake news algorithm: calculate expectation vectors curlyE  
    curlyEs = Array{Float64}(undef, (T, size(o_ss)...))

    # initialize with beginning-of-period expectation of steady-state policy
    curlyEs[1, [Colon() for i ∈ 2:ndims(curlyEs)]...] = demean(expectation(law_of_motion[1], o_ss))
    for t in 2:T
        # demean so that curlyEs converge to zero (in theory no effect but numerically better)
        curlyEs[t, [Colon() for i ∈ 2:ndims(curlyEs)]...] = demean(expectation(law_of_motion, curlyEs[t-1, [Colon() for i ∈ 2:ndims(curlyEs)]...]))
    end
    return curlyEs
end

function build_F(curlyYs, curlyDs, curlyEs)
    # Part 3 of fake news algorithm: build fake news matrix from curlyY, curlyD, curlyE 
    T = size(curlyDs, 1)
    Tpost = size(curlyEs, 1) - T + 2
    F = Array{Float64}(undef, Tpost + T - 1, T)
    F[1,:] = curlyYs
    F[2:end,:] = reshape(curlyEs, Tpost + T - 2, :) * transpose(reshape(curlyDs, T, :)) 
    return F
end

# J_from_F
function J_from_F(F)
    # Part 4 of fake news algorithm: recursively build Jacobian from fake news matrix 
    J = copy(F)
    for t in 2:size(J,2)
        J[2:end, t] += J[1:end-1,t-1]
    end
    return J
end

function backward_step_fakenews(block::HetBlock, din_dict, output_list, differentiable_backward_fun, differentiable_hetoutput, law_of_motion, exog; maybe_exog_shock = false)
    # Support for part 1 of fake news algorithm: single backward step in response to shock 
    Dbeg, D = law_of_motion[1].Dss, law_of_motion[2].Dss

    # shock perturbs outputs
    shocked_outputs = diff(differentiable_backward_fun, din_dict)
    curlyV = Dict{String,Any}(string(k) => expectation(law_of_motion[1], shocked_outputs[string(k)]) for k ∈ block.backward)

    # if there may be a shock to exogenous processes, figure out what it is
    shocks_to_exog = maybe_exog_shock ? [get(din_dict, k, nothing) for k ∈ block.exogenous] : nothing

    # perturbation to exog and outputs--outputs affect distribution tmrw
    policy_shock = [shocked_outputs[k] for k ∈ block.policy]
    if length(policy_shock) == 1
        policy_shock = policy_shock[1]
    end
    curlyD = forward_shock(law_of_motion, [shocks_to_exog, policy_shock])

    # and also affect aggregate outcomes today
    if !isnothing(differentiable_hetoutput) && !isempty(output_list ∩ differentiable_hetoutput.outputs)
        merge!(shocked_outputs, diff(differentiable_hetoutput, Dict(shocked_outputs..., din_dict...), outputs=(differentiable_hetoutput.outputs ∩ output_list)))
    end
    curlyY = Dict{String,Any}(string(k) => dot(D, shocked_outputs[string(k)]) for k ∈ output_list)

    # add effects from perturbation to exog on beginning-of-period expectations in curlyV and curlyY
    if maybe_exog_shock
        for k ∈ keys(curlyV)
            shock = expectation_shock(exog[k], shocks_to_exog)
            if !isnothing(shock)
                curlyV[k] += shock
            end
        end
        for k ∈ keys(curlyY)
            shock = expectation_shock(exog[k], shocks_to_exog)
            if !isnothing(shock)
                curlyY[k] += dot(Dbeg, shock)
            end
        end
    end
    return curlyV, curlyD, curlyY
end

function jac_backward_prelim(block::HetBlock, ss, h, exog, twosided)
    # Support for part 1 of fake news algorithm: preload differentiable functions
    differentiable_hetinputs = nothing
    if !isnothing(block.hetinputs)
        # Hetinputs follow the same differencing rule as everything else in the block:
        # one-sided by default, two-sided when `twosided=true`.
        #
        # The Python toolbox carries the comment "always use two-sided differentiation
        # for hetinputs", but its combined-function `diff` defaults to one-sided, so in
        # practice Python differentiates hetinputs (and hetoutputs) one-sided whatever
        # is requested. One-sided here by default therefore reproduces Python exactly;
        # `twosided=true` makes the WHOLE household Jacobian two-sided (backward step,
        # hetinputs and hetoutputs), which Python's flag does not do.
        differentiable_hetinputs = differentiable(block.hetinputs, ss; h=h, twosided=twosided)
    end

    differentiable_hetoutputs = nothing
    if !isnothing(block.hetoutputs)
        differentiable_hetoutputs = differentiable(block.hetoutputs, ss; h=h, twosided=twosided)
    end

    ss = copy(ss)
    for k ∈ block.backward
        ss[k * "_p"] = expectation(exog, ss[k])
    end
    differentiable_backward_fun = differentiable(block.backward_fun, ss; h=h, twosided=twosided)

    return differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs
end

function process_hetinputs_hetoutputs(block::HetBlock, hetinputs, hetoutputs; tocopy=true)
    if tocopy
        block = deepcopy(block) # deepcopy recursive
    end

    inputs = copy(block.original_inputs)
    outputs = copy(block.original_outputs)
    internals = copy(block.original_internals)

    if !isnothing(hetoutputs)
        inputs = inputs ∪ setdiff(hetoutputs.inputs, block.backward_fun.outputs ∪ OrderedSet{String}(["D"]))
        outputs = outputs ∪ OrderedSet{String}(uppercase(o) for o ∈ hetoutputs.outputs)
        block.M_outputs = Bijection(Dict(o => uppercase(o) for o ∈ hetoutputs.outputs)) * block.original_M_outputs
        internals = internals ∪ hetoutputs.outputs
    end
    if !isnothing(hetinputs)
        inputs = inputs ∪ hetinputs.inputs
        inputs = setdiff(inputs, hetinputs.outputs)
        internals = internals ∪ hetinputs.outputs
    end
    block.inputs = inputs
    block.outputs = outputs
    block.internals = internals

    block.hetinputs = hetinputs
    block.hetoutputs = hetoutputs

    return block
end

# add_hetinputs
function add_hetinputs(block::HetBlock, functions)
    if isnothing(block.hetinputs)
        return process_hetinputs_hetoutputs(block, CombinedExtendedFunction(functions), block.hetoutputs)
    else
        return process_hetinputs_hetoutputs(block, add(block.hetinputs, functions), block.hetoutputs)
    end
end

# remove_hetinputs
function remove_hetinputs(block::HetBlock, names)
    return process_hetinputs_hetoutputs(block, remove(block.hetinputs, names), block.hetoutputs)
end

# add_hetoutputs
function add_hetoutputs(block::HetBlock, functions)
    if isnothing(block.hetoutputs)
        return process_hetinputs_hetoutputs(block, block.hetinputs, CombinedExtendedFunction(functions))
    else
        return process_hetinputs_hetoutputs(block, block.hetinputs, add(block.hetoutputs, functions))
    end
end

# remove_hetoutputs
function remove_hetinputs_hetoutputs(block::HetBlock, names)
    return process_hetinputs_hetoutputs(block, block.hetinputs, remove(block.hetoutputs, names))
end

# update_with_hetinputs
function update_with_hetinputs(block::HetBlock, d)
    if !isnothing(block.hetinputs)
        return merge(d, block.hetinputs(d))
    else
        return d
    end
end

# update_with_hetoutputs
function update_with_hetoutputs(block::HetBlock, d)
    if !isnothing(block.hetoutputs)
        return merge(d, block.hetoutputs(d))
    else
        return d
    end
end

function extract_ss_dict(block::HetBlock, ss)
    if ss isa SteadyStateDict
        ssnew = copy(ss.toplevel)
        if haskey(ss.internals, block.name)
            ssnew = merge(ssnew, ss.internals[block.name])
            end
        return ssnew
    else
        return copy(ss)
    end
end

# initialize_backward
function initialize_backward(block::HetBlock, ss)
    if !all(haskey(ss, k) for k ∈ block.backward)
        return merge(ss, block.backward_init(ss))
    else
        return ss
    end
end

# make_exog_law_of_motion
function make_exog_law_of_motion(block::HetBlock, d)
    return CombinedTransition([Markov(d[k],i) for (i,k) in enumerate(block.exogenous)])
end

# make_endog_law_of_motion
function make_endog_law_of_motion(block::HetBlock, d; monotonic=false)
    pol = collect(block.policy)      # ordered sets are not integer-indexable in recent OrderedCollections
    if length(pol) == 1
        return lottery_1d(d[pol[1]], d[pol[1] * "_grid"]; monotonic=monotonic)
    else
        return lottery_2d(d[pol[1]], d[pol[2]], d[pol[1] * "_grid"], d[pol[2] * "_grid"]; monotonic=monotonic)
    end
end

# =============================================================================
# Nonlinear perfect-foresight transition of the household block
#
#   backward:  starting from the steady-state continuation value at date T, solve
#              the household problem back to date 0 along the given input paths;
#   forward:   starting from the steady-state distribution at date 0, push the
#              distribution through the date-t policies and aggregate.
#
# The terminal distribution is an *outcome* (it is not forced back to steady
# state); see `terminal_distribution_distance` for the corresponding horizon check.
# =============================================================================

# Names that can be requested through `internals`.
const _HET_DIST_INTERNALS = ("D", "Dbeg", "Dbeg_terminal")

function _impulse_nonlinear(b::HetBlock, ssin, inputs; outputs, internals=String[], ss_initial=nothing, monotonic=false)
    ss = extract_ss_dict(b, ssin)
    haskey(ss, "Dbeg") || error("Steady state passed to '$(b.name)' has no distribution: compute it with steady_state/solve_steady_state first.")
    if !isnothing(ss_initial)
        init = ss_initial isa SteadyStateDict ? ss_initial.internals[b.name] : ss_initial
        ss["Dbeg"] = init["Dbeg"]            # start from another economy's distribution
    end

    T = inputs.T
    internals = OrderedSet{String}(string(k) for k in internals)
    out_names = OrderedSet{String}(string(o) for o in outputs)
    out_lower = inv(b.M_outputs) * out_names

    indiv = OrderedSet{String}(string(k) for k in out_lower)
    for k in internals
        k in _HET_DIST_INTERNALS || push!(indiv, k)
    end
    for k in b.policy
        push!(indiv, k)
    end
    for k in indiv
        haskey(ss, k) || error("'$k' is not an individual-level outcome of block '$(b.name)'.")
    end

    individual_paths, exog_path = backward_nonlinear(b, ss, inputs, indiv)
    D_path, Dbeg_path, Dbeg_terminal = forward_nonlinear(b, ss, individual_paths, exog_path, monotonic)

    # aggregate outcomes, reported as deviations from steady state
    aggregates = Dict{String,Any}()
    for (o, lo) in zip(out_names, out_lower)
        ss_val = dot(ss["D"], ss[lo])
        aggregates[o] = fast_aggregate(D_path, individual_paths[lo]) .- ss_val
    end

    # requested internals, also as deviations from steady state (time is dimension 1)
    internals_dict = Dict{String,Any}()
    _lead(x) = reshape(x, 1, size(x)...)
    for k in internals
        if k == "D"
            internals_dict[k] = D_path .- _lead(ss["D"])
        elseif k == "Dbeg"
            internals_dict[k] = Dbeg_path .- _lead(ss["Dbeg"])
        elseif k == "Dbeg_terminal"
            internals_dict[k] = Dbeg_terminal .- ss["Dbeg"]
        else
            internals_dict[k] = individual_paths[k] .- _lead(ss[k])
        end
    end
    int = isempty(internals_dict) ? Dict{String,Any}() : Dict{String,Any}(b.name => internals_dict)
    return ImpulseDict(aggregates; internals=int, T=T)
end

# Solve the household problem backward along the input paths.
# Returns individual-level paths (time is dimension 1) and the exogenous law of
# motion in force at each date.
function backward_nonlinear(b::HetBlock, ss, inputs, toreturn)
    T = inputs.T
    individual_paths = Dict{String,Any}(string(k) => Array{Float64}(undef, T, size(ss[string(k)])...) for k in toreturn)

    backdict = copy(ss)
    exog = make_exog_law_of_motion(b, backdict)
    exog_path = Vector{Any}(undef, T)

    input_names = OrderedSet{String}(string(k) for k in keys(inputs.toplevel))
    # only the hetinput functions downstream of a moving input need re-evaluation
    het_fns = isnothing(b.hetinputs) ? [] :
              _filter(b.hetinputs, collect(values(b.hetinputs.functions)), input_names)
    need_hetoutputs = !isnothing(b.hetoutputs) && any(k in b.hetoutputs.outputs for k in toreturn)

    for t in T:-1:1
        # continuation values: expectation under next period's exogenous transition
        for k in b.backward
            backdict[k * "_p"] = expectation(exog, backdict[k])
            delete!(backdict, k)
        end

        for (k, v) in inputs.toplevel
            backdict[string(k)] = ss[string(k)] .+ v[t]
        end
        for g in het_fns
            merge!(backdict, g(backdict))
        end
        exog = make_exog_law_of_motion(b, backdict)

        merge!(backdict, b.backward_fun(backdict))
        if need_hetoutputs
            for (k, v) in b.hetoutputs(backdict)
                k in b.hetoutputs.outputs && (backdict[k] = v)
            end
        end

        for (k, path) in individual_paths
            selectdim(path, 1, t) .= backdict[k]
        end
        exog_path[t] = exog
    end

    return individual_paths, exog_path
end

# Push the distribution forward from the steady state through the date-t policies.
# Returns D (after the exogenous transition, the one used for aggregation), Dbeg
# (beginning of period) and the beginning-of-period distribution at date T.
function forward_nonlinear(b::HetBlock, ss, individual_paths, exog_path, monotonic)
    T = length(exog_path)
    Dbeg = ss["Dbeg"]
    D_path = Array{Float64}(undef, T, size(Dbeg)...)
    Dbeg_path = Array{Float64}(undef, T, size(Dbeg)...)

    d = Dict{String,Any}()
    for k in b.policy
        d[k * "_grid"] = ss[k * "_grid"]
    end
    for t in 1:T
        for k in b.policy
            d[k] = collect(selectdim(individual_paths[k], 1, t))
        end
        endog = make_endog_law_of_motion(b, d; monotonic=monotonic)

        selectdim(Dbeg_path, 1, t) .= Dbeg
        D = forward(exog_path[t], Dbeg)
        selectdim(D_path, 1, t) .= D
        Dbeg = forward(endog, D)
    end
    return D_path, Dbeg_path, Dbeg
end
