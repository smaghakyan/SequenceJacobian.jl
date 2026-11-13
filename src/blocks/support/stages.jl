# stages.jl: the stages a StageBlock is made of.
# Julia version of Python SSJ blocks/support/stages.py.
#
# A stage maps end-of-stage "backward" objects (value functions and their
# derivatives) into beginning-of-stage ones, and produces a law of motion that
# moves the distribution from the beginning to the end of the stage.
#
#   Exogenous / ExogenousMaker : Markov transition along one dimension (with `by = j`: a
#                                different matrix for each position in dimension j; with
#                                `joint = true`: one transition for dimensions 1 and 2 together)
#   LogitChoice                : discrete choice with type-1 extreme-value taste shocks;
#                                the choice sets the value of one discrete dimension
#   Continuous1D, Continuous2D : continuous choice of one or two endogenous states
#                                (the last one or two dimensions of every array)
#
# Conventions (as in Python, with 1-based dimensions):
#   * all arrays of a StageBlock share one shape; `index` is the dimension a stage acts on;
#   * a stage function takes the end-of-stage backward objects under their own names
#     (no "_p" suffix) and returns the beginning-of-stage ones under the same names;
#   * the flow-utility function of a LogitChoice returns ONE array of shape
#     (n_choices, size of the state space) or anything that broadcasts to it,
#     with -Inf for choices that are not available.

using OrderedCollections: OrderedSet

abstract type Stage end

_stage_set(x) = OrderedSet{String}(string(k) for k in make_tuple(x))
_stage_fun(f) = f isa AbstractExtendedFunction ? f : ExtendedFunction(f)
_stage_hetoutputs(h) = isnothing(h) ? nothing :
                       (h isa AbstractCombinedExtendedFunction ? h : CombinedExtendedFunction(collect(make_tuple(h))))

# Backward step that also evaluates hetoutputs and separates backward objects,
# reported outcomes and (optionally) the law of motion.
function backward_step_separate(stage::Stage, inputs::AbstractDict; lawofmotion=false, hetoutputs=false)
    lom = nothing
    outputs = backward_step(stage, inputs; lawofmotion=lawofmotion)
    if lawofmotion
        outputs, lom = outputs
    end
    backward_out = Dict{String,Any}(k => outputs[k] for k in stage.backward_outputs)
    report_out   = Dict{String,Any}(k => outputs[k] for k in stage.original_report)
    if hetoutputs && !isnothing(stage.hetoutputs)
        res = stage.hetoutputs(merge(Dict{String,Any}(inputs), outputs))
        for k in stage.hetoutputs.outputs
            report_out[k] = res[k]
        end
    end
    return lawofmotion ? ((backward_out, report_out), lom) : (backward_out, report_out)
end

function process_hetoutputs!(stage::Stage, hetoutputs)
    stage.inputs = copy(stage.original_inputs)
    stage.report = copy(stage.original_report)
    if !isnothing(hetoutputs)
        union!(stage.inputs, setdiff(hetoutputs.inputs, union(stage.report, stage.backward_outputs)))
        union!(stage.report, hetoutputs.outputs)
    end
    stage.hetoutputs = hetoutputs
    return stage
end

function add_hetoutputs(stage::Stage, functions)
    new = deepcopy(stage)
    ho = isnothing(stage.hetoutputs) ? CombinedExtendedFunction(collect(make_tuple(functions))) :
                                       add(stage.hetoutputs, functions)
    return process_hetoutputs!(new, ho)
end

# ------------------------------------------------------------------------------
# Laws of motion used by the stages. Every law of motion `lom` supports
#   forward(lom, D)       distribution at the end of the stage
#   expectation(lom, X)   beginning-of-stage expectation of an end-of-stage array
# and every perturbed law of motion `dlom` supports
#   forward_shock(dlom, D)   first-order change in the end-of-stage distribution
# Markov and PolicyLottery1D/2D come from het_support.jl.
# ------------------------------------------------------------------------------

struct DiscreteChoice <: Transition
    P::Array{Float64}       # P[d, state...]: probability of choice d
    index::Int              # dimension of the state space set by the choice
    DiscreteChoice(P::AbstractArray, index::Int) = new(Array{Float64}(P), index)
end
# P with the choice and the state it sets exchanged (not stored: a nonlinear transition
# keeps one law of motion per date)
function _choice_transpose(dc::DiscreteChoice)
    perm = collect(1:ndims(dc.P))
    perm[1], perm[dc.index+1] = perm[dc.index+1], perm[1]
    return permutedims(dc.P, perm)
end
forward(dc::DiscreteChoice, D)       = batch_multiply_ith_dimension(dc.P, dc.index, D)
expectation(dc::DiscreteChoice, X)   = batch_multiply_ith_dimension(_choice_transpose(dc), dc.index, X)
forward_shock(dc::DiscreteChoice, D) = batch_multiply_ith_dimension(dc.P, dc.index, D)

# Markov matrix along dimension `index` that differs by the position in dimension `by`:
# Pi[j, x, x'] is the transition matrix for the entries with index j in dimension `by`.
struct MarkovBy <: Transition
    Pi::Array{Float64,3}
    index::Int
    by::Int
    function MarkovBy(Pi::AbstractArray, index::Int, by::Int)
        ndims(Pi) == 3 || error("MarkovBy: the transition array must have three dimensions (group, from, to)")
        index != by || error("MarkovBy: `index` and `by` must be different dimensions")
        new(Array{Float64,3}(Pi), index, by)
    end
end
function _apply_markov_by(m::MarkovBy, X, transposed::Bool)
    out = Array{Float64}(undef, size(X))
    idx = m.index > m.by ? m.index - 1 : m.index           # position of `index` once `by` is fixed
    for j in 1:size(m.Pi, 1)
        M = transposed ? permutedims(m.Pi[j, :, :]) : m.Pi[j, :, :]
        selectdim(out, m.by, j) .= multiply_ith_dimension(M, idx, collect(selectdim(X, m.by, j)))
    end
    return out
end
forward(m::MarkovBy, D)     = _apply_markov_by(m, D, true)
expectation(m::MarkovBy, X) = _apply_markov_by(m, X, false)

# Markov transition on the first two dimensions jointly: Pi[i, j, i', j'] is the probability of
# moving from position (i, j) to position (i', j').
struct MarkovJoint <: Transition
    M::Matrix{Float64}          # rows: (i, j), columns: (i', j'), both in column-major order
    n1::Int
    n2::Int
    function MarkovJoint(Pi::AbstractArray)
        ndims(Pi) == 4 || error("MarkovJoint: the transition array must have four dimensions (from 1, from 2, to 1, to 2)")
        n1, n2 = size(Pi, 1), size(Pi, 2)
        (size(Pi, 3) == n1 && size(Pi, 4) == n2) || error("MarkovJoint: the transition array must have size (n1, n2, n1, n2)")
        new(Matrix{Float64}(reshape(Pi, n1*n2, n1*n2)), n1, n2)
    end
end
function _apply_markov_joint(m::MarkovJoint, X, transposed::Bool)
    (size(X, 1) == m.n1 && size(X, 2) == m.n2) || error("MarkovJoint: the first two dimensions of the array do not match the transition array")
    Xm = reshape(X, m.n1*m.n2, :)
    return reshape(transposed ? transpose(m.M)*Xm : m.M*Xm, size(X))
end
forward(m::MarkovJoint, D)     = _apply_markov_joint(m, D, true)
expectation(m::MarkovJoint, X) = _apply_markov_joint(m, X, false)

# perturbation of a Markov matrix
struct ShockedMarkov <: Transition
    dPi_T
    index::Int
end
forward_shock(sm::ShockedMarkov, D) = multiply_ith_dimension(sm.dPi_T, sm.index, D)

struct ShockedPolicyLottery1D <: Transition
    i; pi_shock; flatshape; shape
end
forward_shock(s::ShockedPolicyLottery1D, D) =
    reshape(forward_policy_shock_1d(reshape(D, s.flatshape), s.i, s.pi_shock), s.shape)

struct ShockedPolicyLottery2D <: Transition
    i1; i2; pi1; pi2; pi_shock1; pi_shock2; flatshape; shape
end
forward_shock(s::ShockedPolicyLottery2D, D) =
    reshape(forward_policy_shock_2d(reshape(D, s.flatshape), s.i1, s.i2, s.pi1, s.pi2,
                                    s.pi_shock1, s.pi_shock2), s.shape)

# ------------------------------------------------------------------------------
# Continuous1D
# ------------------------------------------------------------------------------
mutable struct Continuous1D <: Stage
    f
    policy::String
    name::String
    backward_outputs::OrderedSet{String}
    report::OrderedSet{String}
    inputs::OrderedSet{String}
    original_inputs::OrderedSet{String}
    original_report::OrderedSet{String}
    hetoutputs
end

function Continuous1D(backward, policy, f; name=nothing, hetoutputs=nothing)
    ef   = _stage_fun(f)
    back = _stage_set(backward)
    rep  = OrderedSet{String}(k for k in ef.outputs if !(k in back))
    inp  = OrderedSet{String}(ef.inputs)
    all(k in ef.outputs for k in back) || error("Stage '$(ef.name)': backward variables $(collect(back)) must all be returned (returned: $(collect(ef.outputs)))")
    string(policy) in ef.outputs || error("Stage '$(ef.name)': policy '$policy' is not returned")
    stage = Continuous1D(ef, string(policy), isnothing(name) ? ef.name : string(name), back, rep, inp, copy(inp), copy(rep), nothing)
    return process_hetoutputs!(stage, _stage_hetoutputs(hetoutputs))
end

Base.show(io::IO, s::Continuous1D) = print(io, "<Stage-Continuous1D '$(s.name)' with policy '$(s.policy)'>")

function backward_step(s::Continuous1D, inputs::AbstractDict; lawofmotion=false)
    outputs = s.f(inputs)
    lawofmotion || return outputs
    return outputs, lottery_1d(outputs[s.policy], inputs[s.policy * "_grid"]; monotonic=false)
end

function precompute(s::Continuous1D, ss::AbstractDict, lom; h=1e-4, twosided=false)
    i = reshape(lom.i, lom.shape)
    return (space = lom.grid[i .+ 1] .- lom.grid[i], lom = lom, f = differentiable(s.f, ss; h=h, twosided=twosided))
end

function backward_step_shock(s::Continuous1D, ss::AbstractDict, shocks::AbstractDict, pre)
    outputs = diff(pre.f, shocks)
    dpi = -outputs[s.policy] ./ pre.space
    lom = pre.lom
    return outputs, ShockedPolicyLottery1D(lom.i, reshape(dpi, lom.flatshape), lom.flatshape, lom.shape)
end

# ------------------------------------------------------------------------------
# Continuous2D
# ------------------------------------------------------------------------------
mutable struct Continuous2D <: Stage
    f
    policy::Vector{String}
    name::String
    backward_outputs::OrderedSet{String}
    report::OrderedSet{String}
    inputs::OrderedSet{String}
    original_inputs::OrderedSet{String}
    original_report::OrderedSet{String}
    hetoutputs
end

function Continuous2D(backward, policy, f; name=nothing, hetoutputs=nothing)
    ef   = _stage_fun(f)
    back = _stage_set(backward)
    pol  = [string(k) for k in policy]
    length(pol) == 2 || error("Continuous2D needs exactly two policies")
    rep  = OrderedSet{String}(k for k in ef.outputs if !(k in back))
    inp  = OrderedSet{String}(ef.inputs)
    all(k in ef.outputs for k in back) || error("Stage '$(ef.name)': backward variables $(collect(back)) must all be returned (returned: $(collect(ef.outputs)))")
    all(k in ef.outputs for k in pol) || error("Stage '$(ef.name)': policies $pol must be returned")
    stage = Continuous2D(ef, pol, isnothing(name) ? ef.name : string(name), back, rep, inp, copy(inp), copy(rep), nothing)
    return process_hetoutputs!(stage, _stage_hetoutputs(hetoutputs))
end

Base.show(io::IO, s::Continuous2D) = print(io, "<Stage-Continuous2D '$(s.name)' with policies $(s.policy)>")

function backward_step(s::Continuous2D, inputs::AbstractDict; lawofmotion=false)
    outputs = s.f(inputs)
    lawofmotion || return outputs
    return outputs, lottery_2d(outputs[s.policy[1]], outputs[s.policy[2]],
                               inputs[s.policy[1] * "_grid"], inputs[s.policy[2] * "_grid"])
end

function precompute(s::Continuous2D, ss::AbstractDict, lom; h=1e-4, twosided=false)
    i1 = reshape(lom.i1, lom.shape); i2 = reshape(lom.i2, lom.shape)
    return (space1 = lom.grid1[i1 .+ 1] .- lom.grid1[i1], space2 = lom.grid2[i2 .+ 1] .- lom.grid2[i2],
            lom = lom, f = differentiable(s.f, ss; h=h, twosided=twosided))
end

function backward_step_shock(s::Continuous2D, ss::AbstractDict, shocks::AbstractDict, pre)
    outputs = diff(pre.f, shocks)
    lom  = pre.lom
    dpi1 = reshape(-outputs[s.policy[1]] ./ pre.space1, lom.flatshape)
    dpi2 = reshape(-outputs[s.policy[2]] ./ pre.space2, lom.flatshape)
    return outputs, ShockedPolicyLottery2D(lom.i1, lom.i2, lom.pi1, lom.pi2, dpi1, dpi2, lom.flatshape, lom.shape)
end

# ------------------------------------------------------------------------------
# Exogenous (and its maker: the backward variables are those of the next stage)
# ------------------------------------------------------------------------------
struct ExogenousMaker
    markov_name::String
    index::Int
    name::String
    hetoutputs
    by::Int                 # 0: one matrix for all; j > 0: one matrix for each position in dimension j
    joint::Bool             # true: one transition Pi[i, j, i', j'] for dimensions 1 and 2 together (index must be 1)
end
ExogenousMaker(markov_name, index; name=nothing, hetoutputs=nothing, by=0, joint=false) =
    ExogenousMaker(string(markov_name), Int(index), isnothing(name) ? "exog_$(markov_name)" : string(name), hetoutputs, Int(by), Bool(joint))
ExogenousMaker(markov_name, index, name; hetoutputs=nothing, by=0, joint=false) =
    ExogenousMaker(markov_name, index; name=name, hetoutputs=hetoutputs, by=by, joint=joint)

make_stage(em::ExogenousMaker, backward) = Exogenous(em.markov_name, em.index, em.name, backward; hetoutputs=em.hetoutputs, by=em.by, joint=em.joint)

mutable struct Exogenous <: Stage
    markov_name::String
    index::Int
    by::Int
    joint::Bool
    name::String
    backward_outputs::OrderedSet{String}
    report::OrderedSet{String}
    inputs::OrderedSet{String}
    original_inputs::OrderedSet{String}
    original_report::OrderedSet{String}
    hetoutputs
end

function Exogenous(markov_name, index, name, backward; hetoutputs=nothing, by=0, joint=false)
    back = _stage_set(collect(backward))
    rep  = OrderedSet{String}()
    inp  = union(back, OrderedSet{String}([string(markov_name)]))
    if joint
        (Int(index) == 1 && Int(by) == 0) || error("Exogenous stage '$(name)': joint = true needs index = 1 and no `by` (the transition acts on dimensions 1 and 2)")
    end
    stage = Exogenous(string(markov_name), Int(index), Int(by), Bool(joint), string(name), back, rep, inp, copy(inp), copy(rep), nothing)
    return process_hetoutputs!(stage, _stage_hetoutputs(hetoutputs))
end

_exog_lom(s::Exogenous, Pi) = s.joint ? MarkovJoint(Pi) : (s.by == 0 ? Markov(Matrix{Float64}(Pi), s.index) : MarkovBy(Pi, s.index, s.by))

Base.show(io::IO, s::Exogenous) = print(io, "<Stage-Exogenous '$(s.name)' with Markov matrix '$(s.markov_name)'>")

function backward_step(s::Exogenous, inputs::AbstractDict; lawofmotion=false)
    Pi = _exog_lom(s, inputs[s.markov_name])
    outputs = Dict{String,Any}(k => expectation(Pi, inputs[k]) for k in s.backward_outputs)
    lawofmotion || return outputs
    return outputs, Pi
end

precompute(s::Exogenous, ss::AbstractDict, lom; h=1e-4, twosided=false) = nothing

function backward_step_shock(s::Exogenous, ss::AbstractDict, shocks::AbstractDict, pre)
    Pi = _exog_lom(s, ss[s.markov_name])
    outputs = Dict{String,Any}(k => expectation(Pi, shocks[k]) for k in s.backward_outputs if haskey(shocks, k))
    if haskey(shocks, s.markov_name)
        (s.by == 0 && !s.joint) || error("Stage '$(s.name)': a shock to a transition array that differs by group (by) or acts on two dimensions (joint) is not implemented")
        dPi = Matrix{Float64}(shocks[s.markov_name])
        for k in s.backward_outputs
            term = multiply_ith_dimension(dPi, s.index, ss[k])
            outputs[k] = haskey(outputs, k) ? outputs[k] .+ term : term
        end
        return outputs, ShockedMarkov(Matrix(transpose(dPi)), s.index)
    end
    return outputs, nothing
end

# ------------------------------------------------------------------------------
# LogitChoice
# ------------------------------------------------------------------------------
mutable struct LogitChoice <: Stage
    f                       # flow utility (ExtendedFunction with one output) or nothing
    index::Int
    value::String
    backward::OrderedSet{String}
    taste_shock_scale::String
    name::String
    backward_outputs::OrderedSet{String}
    report::OrderedSet{String}
    inputs::OrderedSet{String}
    original_inputs::OrderedSet{String}
    original_report::OrderedSet{String}
    hetoutputs
end

function LogitChoice(value, backward, index, taste_shock_scale; f=nothing, name=nothing, hetoutputs=nothing)
    ef = isnothing(f) ? nothing : _stage_fun(f)
    (!isnothing(ef) && length(ef.outputs) != 1) && error("Flow utility function '$(ef.name)' must return exactly one output (it returns $(collect(ef.outputs)))")
    nm   = isnothing(name) ? (isnothing(ef) ? "logit_$(value)" : ef.name) : string(name)
    back = _stage_set(backward)
    back_out = union(back, OrderedSet{String}([string(value)]))
    rep  = OrderedSet{String}()
    inp  = union(back, OrderedSet{String}([string(value), string(taste_shock_scale)]))
    isnothing(ef) || union!(inp, ef.inputs)
    stage = LogitChoice(ef, Int(index), string(value), back, string(taste_shock_scale), nm,
                        back_out, rep, inp, copy(inp), copy(rep), nothing)
    return process_hetoutputs!(stage, _stage_hetoutputs(hetoutputs))
end

Base.show(io::IO, s::LogitChoice) = print(io, "<Stage-Discrete '$(s.name)'>")

# value of each choice d at each beginning-of-stage state: put the choice in dimension 1
function _choice_array(X, index::Int)
    Y = reshape(X, 1, size(X)...)
    perm = collect(1:ndims(Y))
    perm[1], perm[index+1] = perm[index+1], perm[1]
    return permutedims(Y, perm)
end

function backward_step(s::LogitChoice, inputs::AbstractDict; lawofmotion=false)
    V_next = inputs[s.value]
    V = _choice_array(V_next, s.index)
    flow_u = isnothing(s.f) ? zeros(size(V, 1), size(V_next)...) : first(values(s.f(inputs)))
    V = flow_u .+ V
    P, EV = logit_choice(V, inputs[s.taste_shock_scale])
    lom = DiscreteChoice(P, s.index)
    outputs = Dict{String,Any}(k => expectation(lom, inputs[k]) for k in s.backward)
    outputs[s.value] = dropdims(EV, dims=1)
    lawofmotion || return outputs
    return outputs, lom
end

function precompute(s::LogitChoice, ss::AbstractDict, lom; h=1e-4, twosided=false)
    return (f = isnothing(s.f) ? nothing : differentiable(s.f, ss; h=h, twosided=twosided), lom = lom)
end

function backward_step_shock(s::LogitChoice, ss::AbstractDict, shocks::AbstractDict, pre)
    lom = pre.lom
    P = lom.P
    dV = haskey(shocks, s.value) ? _choice_array(shocks[s.value], s.index) : zeros(1)
    if !isnothing(pre.f)
        dflow = first(values(diff(pre.f, shocks)))
        dflow = map(x -> isfinite(x) ? x : 0.0, dflow)      # -Inf - (-Inf) = NaN: these choices do not exist
        dV = dflow .+ dV
    end
    dV  = dV .+ zeros(size(P))
    dEV = sum(P .* dV, dims=1)                               # envelope
    dP  = P .* (dV .- dEV) ./ ss[s.taste_shock_scale]
    dlom = DiscreteChoice(dP, s.index)

    douts = Dict{String,Any}(s.value => dropdims(dEV, dims=1))
    for k in s.backward
        douts[k] = expectation(dlom, ss[k])
        haskey(shocks, k) && (douts[k] = douts[k] .+ expectation(lom, shocks[k]))
    end
    return douts, dlom
end

# ------------------------------------------------------------------------------
# Turn a list of stages and stage makers into stages
# ------------------------------------------------------------------------------
function make_all_into_stages(stages)
    stages = Any[s for s in stages]
    ifirst = findfirst(s -> s isa Stage, stages)
    isnothing(ifirst) && error("No full-fledged stage supplied to StageBlock (only makers).")
    for i in ifirst-1:-1:1
        stages[i] isa Stage || (stages[i] = make_stage(stages[i], stages[i+1].backward_outputs))
    end
    for i in length(stages):-1:ifirst+1
        stages[i] isa Stage || (stages[i] = make_stage(stages[i], stages[mod1(i+1, length(stages))].backward_outputs))
    end
    return Stage[s for s in stages]
end
