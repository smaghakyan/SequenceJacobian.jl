# diagnostics.jl — validation and household-statistics tools.
#
# These follow the checks used in Yuxuan Zhao's sequence-space notebooks (RBC,
# Krusell-Smith, one- and two-asset HANK), written here for arbitrary blocks:
#
#   jacobian_direct              brute-force block Jacobian from nonlinear transitions
#   fake_news_error              fake-news Jacobian vs the brute-force one
#   steady_path_error            residuals when the steady state is fed through the
#                                nonlinear transition (catches timing mistakes)
#   distribution_distance        ||D_t - D_ss|| along a path, incl. the terminal date
#   terminal_distribution_distance
#   channel_decomposition        GE response of a household outcome split by household input
#   mpc_statistics               finite-transfer MPCs (impact and cumulative) and group means
#   weighted_quantile

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

# All heterogeneous-agent blocks inside a (possibly nested) block.
function het_blocks(b::Block)
    found = HetBlock[]
    _collect_het!(found, b)
    return found
end
function _collect_het!(found, b)
    if b isa HetBlock
        push!(found, b)
    elseif hasproperty(b, :blocks)
        for k in b.blocks; _collect_het!(found, k); end
    elseif hasproperty(b, :block)
        _collect_het!(found, b.block)
    end
    return found
end

function _the_het_block(model::Block, block)
    block isa HetBlock && return block
    hets = het_blocks(model)
    if isnothing(block)
        length(hets) == 1 || error("Model has $(length(hets)) heterogeneous-agent blocks: say which one with `block=`.")
        return hets[1]
    end
    for h in hets
        h.name == string(block) && return h
    end
    error("No heterogeneous-agent block named '$block' in '$(model.name)'.")
end

"""
    weighted_quantile(values, weights, probability)

First ordered value whose cumulative weight reaches `probability` (in [0, 1]).
"""
function weighted_quantile(values, weights, probability)
    length(values) == length(weights) || throw(ArgumentError("values and weights must have the same length"))
    0.0 <= probability <= 1.0 || throw(ArgumentError("probability must lie in [0,1]"))
    v = vec(collect(values)); w = vec(collect(weights))
    total = sum(w)
    total > 0 || throw(ArgumentError("weights must have positive mass"))
    order = sortperm(v)
    cumulative = cumsum(w[order]) ./ total
    index = min(searchsortedfirst(cumulative, probability), length(order))
    return v[order[index]]
end

# ------------------------------------------------------------------------------
# Brute-force Jacobian and fake-news validation
# ------------------------------------------------------------------------------

"""
    jacobian_direct(block, ss, inputs, outputs; T=10, h=1e-6)

Brute-force sequence-space Jacobian of `block`: perturb each input at each date by
`h`, run the full nonlinear transition, and difference against the unperturbed
transition. Costs `T` transitions per input, so keep `T` small; it exists to check
`jacobian` (the fake-news algorithm for heterogeneous-agent blocks), not to replace it.
"""
function jacobian_direct(b::Block, ss, inputs, outputs=nothing; T::Int=10, h=1e-6, kwargs...)
    inputs = make_ordered_set(inputs)
    outs = isnothing(outputs) ? setdiff(b.outputs, _vector_valued(ss)) : make_ordered_set(outputs)
    J = Dict{String,Any}(o => Dict{String,Any}() for o in outs)
    for i in inputs
        base = impulse_nonlinear(b, ss, ImpulseDict(Dict{String,Any}(i => zeros(T)); T=T); outputs=outs, kwargs...)
        cols = Dict{String,Matrix{Float64}}(o => zeros(T, T) for o in outs)
        for s in 1:T
            dx = zeros(T); dx[s] = h
            out = impulse_nonlinear(b, ss, ImpulseDict(Dict{String,Any}(i => dx); T=T); outputs=outs, kwargs...)
            for o in outs
                cols[o][:, s] = (get(out, o) .- get(base, o)) ./ h
            end
        end
        for o in outs
            J[o][i] = cols[o]
        end
    end
    return JacobianDict(J; outputs=outs, inputs=inputs, name=b.name, T=T)
end

"""
    fake_news_error(block, ss, inputs, outputs; T=10, h=1e-6, h_direct=h, twosided=false)

Largest absolute difference between the block's `jacobian` and `jacobian_direct`,
overall (`max_error`) and per output/input pair (`errors`). For a heterogeneous-agent
block this validates the timing and the diagonal recursion of the fake-news algorithm.

`h` and `twosided` are passed to `jacobian`; `h_direct` is the step of the brute-force
perturbation. Two uses:

  * correctness of the algorithm: same small step on both sides (the default);
  * accuracy of the production settings: `h=1e-4, h_direct=1e-6`, with and without
    `twosided=true`. One-sided differences with `h=1e-4` carry an error of order
    1e-4 to 1e-3 in relative terms for strongly nonlinear household problems;
    `twosided=true` removes most of it at twice the cost.
"""
function fake_news_error(b::Block, ss, inputs, outputs=nothing; T::Int=10, h=1e-6, h_direct=h, twosided::Bool=false)
    inputs = make_ordered_set(inputs)
    outs = isnothing(outputs) ? setdiff(b.outputs, _vector_valued(ss)) : make_ordered_set(outputs)
    Jd = jacobian_direct(b, ss, inputs, outs; T=T, h=h_direct)
    Jf = (b isa HetBlock || b isa StageBlock) ? jacobian(b, ss, inputs, outs; T=T, h=h, twosided=twosided) : jacobian(b, ss, inputs, outs; T=T)
    errors = Dict{String,Float64}()
    for o in outs, i in inputs
        direct = Jd.nesteddict[o][i]
        fast = (haskey(Jf.nesteddict, o) && haskey(Jf.nesteddict[o], i)) ? make_matrix(Jf.nesteddict[o][i], T) : zeros(T, T)
        errors["$(o),$(i)"] = maximum(abs.(direct .- fast))
    end
    return (max_error = isempty(errors) ? 0.0 : maximum(values(errors)), errors = errors, direct = Jd, fake_news = Jf)
end

"""
    steady_path_error(model, ss, unknowns, targets; T=20)

Feed the steady state itself through the nonlinear transition (all inputs constant)
and return the largest absolute deviation of each target from zero, plus the overall
maximum. It should be at the level of the steady-state tolerances; anything larger
means the steady state and the transition equations disagree (usually a timing error
or a steady state that did not converge).
"""
function steady_path_error(model::Block, ss, unknowns, targets; T::Int=20, kwargs...)
    unknowns = make_ordered_set(unknowns); targets = make_ordered_set(targets)
    zero_paths = ImpulseDict(Dict{String,Any}(k => zeros(T) for k in unknowns); T=T)
    out = impulse_nonlinear(model, ss, zero_paths; outputs=targets, ss_initial=nothing, kwargs...)
    errs = Dict{String,Float64}(k => maximum(abs, get(out, k)) for k in targets)
    # blocks whose inputs do not move are skipped by the transition; force every
    # target to be evaluated by also reporting its steady-state level
    levels = Dict{String,Float64}(k => (haskey(ss, k) ? abs(Float64(ss[k])) : NaN) for k in targets)
    return (max_error = maximum(values(errs)), errors = errs, steady_state_levels = levels)
end

# ------------------------------------------------------------------------------
# Distribution diagnostics
# ------------------------------------------------------------------------------

"""
    distribution_distance(model, ss, paths; block=nothing)

Run the nonlinear transition of `model` along `paths` (a Dict or ImpulseDict of
deviations for shocks and, in general equilibrium, the unknowns) and report how far
the household distribution is from its steady state:

  * `path[t]`   = max |D_t − D_ss| for t = 1..T
  * `terminal`  = max |D_{T+1} − D_ss| (beginning-of-period distribution after the last date)

The finite-horizon solution imposes the steady-state *value function* at date T but
not the steady-state *distribution*. If `terminal` is not small, the horizon is too
short for this shock: increase T.
"""
function distribution_distance(model::Block, ss, paths; block=nothing, kwargs...)
    hb = _the_het_block(model, block)
    paths = paths isa ImpulseDict ? paths : ImpulseDict(paths)
    out = impulse_nonlinear(model, ss, paths; internals=Dict{String,Any}(hb.name => ["D", "Dbeg_terminal"]), kwargs...)
    return _distance_from_internals(out, hb.name)
end

function _distance_from_internals(irf::ImpulseDict, name)
    haskey(irf.internals, name) || error("No internals for block '$name': request them with internals=Dict(\"$name\" => [\"D\", \"Dbeg_terminal\"]).")
    int = irf.internals[name]
    haskey(int, "D") || error("The distribution path was not kept: request internals=Dict(\"$name\" => [\"D\", \"Dbeg_terminal\"]).")
    dD = int["D"]
    T = size(dD, 1)
    flat = reshape(dD, T, :)
    path = [maximum(abs, view(flat, t, :)) for t in 1:T]
    terminal = haskey(int, "Dbeg_terminal") ? maximum(abs, int["Dbeg_terminal"]) : NaN
    return (path = path, terminal = terminal)
end

"""
    terminal_distribution_distance(irf, block_name)

max |D_{T+1} − D_ss| from a nonlinear impulse computed with
`internals = Dict(block_name => ["D", "Dbeg_terminal"])`.
"""
terminal_distribution_distance(irf::ImpulseDict, name) = _distance_from_internals(irf, string(name)).terminal

# ------------------------------------------------------------------------------
# Channel decomposition
# ------------------------------------------------------------------------------

"""
    channel_decomposition(model, ss, G, shocks; block=nothing, output="C", direct=String[], T=nothing, Js=Dict())

Split the general-equilibrium response of a household aggregate into the
contributions of the household block's inputs:

    dO = Σ_x J^{O,x} dx,      dx = Σ_z G^{x,z} dz

where `J` is the household block's partial-equilibrium Jacobian, `G` the
general-equilibrium Jacobian from `solve_jacobian` (it must contain the household
inputs as outputs, which is the default), and `shocks` a Dict of shock paths.

Inputs listed in `direct` are summed into `direct` (Kaplan, Moll and Violante call
the liquid-return channel the direct effect); all others go into `indirect`.

Returns a named tuple with `contributions` (Dict input => path), `input_paths`,
`direct`, `indirect`, `total` (from `G`), `reconstructed`, and `closure_error`
(max |total − reconstructed|, which should be at rounding level).
"""
function channel_decomposition(model::Block, ss, G, shocks; block=nothing, output="C", direct=String[], T=nothing, Js=Dict())
    hb = _the_het_block(model, block)
    output = string(output)
    output in hb.outputs || error("'$output' is not an output of household block '$(hb.name)' (outputs: $(collect(hb.outputs))).")
    shocks = Dict{String,Any}(string(k) => v for (k, v) in pairs(shocks))
    isnothing(T) && (T = length(first(values(shocks))))
    gkeys = Set(string(k) for k in keys(G))

    # equilibrium path of every household input that moves
    input_paths = Dict{String,Vector{Float64}}()
    for x in hb.inputs
        path = zeros(T); moves = false
        if haskey(shocks, x)
            path .+= shocks[x]; moves = true
        end
        if x in gkeys
            Gx = G[x]
            for (z, dz) in shocks
                if haskey(Gx, z)
                    path .+= make_matrix(Gx[z], T) * dz; moves = true
                end
            end
        end
        moves && (input_paths[x] = path)
    end
    isempty(input_paths) && error("None of the inputs of '$(hb.name)' responds to the shocks $(collect(keys(shocks))).")

    moving = OrderedSet{String}(sort!(collect(keys(input_paths))))
    J = (haskey(Js, hb.name) && Js[hb.name] isa JacobianDict && moving ⊆ Js[hb.name].inputs && output in Js[hb.name].outputs) ?
        Js[hb.name] : jacobian(hb, ss, moving, [output]; T=T)

    contributions = Dict{String,Vector{Float64}}()
    Jo = J.nesteddict[output]
    for x in moving
        contributions[x] = haskey(Jo, x) ? make_matrix(Jo[x], T) * input_paths[x] : zeros(T)
    end

    direct = [string(d) for d in direct]
    for d in direct
        d in hb.inputs || error("'$d' (listed in `direct`) is not an input of '$(hb.name)'.")
    end
    direct_path = zeros(T); indirect_path = zeros(T)
    for (x, c) in contributions
        x in direct ? (direct_path .+= c) : (indirect_path .+= c)
    end
    reconstructed = direct_path .+ indirect_path

    output in gkeys || error("G has no response for '$output'.")
    total = zeros(T)
    for (z, dz) in shocks
        haskey(G[output], z) && (total .+= make_matrix(G[output][z], T) * dz)
    end
    closure_error = maximum(abs.(total .- reconstructed))
    if closure_error > 1e-6 * max(1.0, maximum(abs.(total)))
        @warn "Channel decomposition does not add up (error $(closure_error)). G is probably missing some household inputs: call solve_jacobian without restricting `outputs`."
    end
    return (contributions = contributions, input_paths = input_paths, direct = direct_path, indirect = indirect_path,
            total = total, reconstructed = reconstructed, closure_error = closure_error, horizon = T, output = output)
end

# ------------------------------------------------------------------------------
# MPC statistics
# ------------------------------------------------------------------------------

# One backward step at the steady state with some household inputs changed.
# Names that the backward function takes directly (including hetinput outputs such as
# an individual transfer array) are overridden after the hetinputs are evaluated.
function _household_step_at_ss(hb::HetBlock, ssd::AbstractDict, shocks::AbstractDict)
    d = copy(ssd)
    exog = make_exog_law_of_motion(hb, d)
    for k in hb.backward
        d[k * "_p"] = expectation(exog, d[k])
        delete!(d, k)
    end
    late = Dict{String,Any}()
    for (k, v) in shocks
        ks = string(k)
        haskey(ssd, ks) || error("'$ks' is not a variable of household block '$(hb.name)'.")
        if ks in hb.backward_fun.inputs
            late[ks] = ssd[ks] .+ v
        else
            d[ks] = ssd[ks] .+ v
        end
    end
    if !isnothing(hb.hetinputs)
        early = OrderedSet{String}(string(k) for k in keys(shocks) if !(string(k) in hb.backward_fun.inputs))
        for g in _filter(hb.hetinputs, collect(values(hb.hetinputs.functions)), early)
            merge!(d, g(d))
        end
    end
    merge!(d, late)
    merge!(d, hb.backward_fun(d))
    if !isnothing(hb.hetoutputs)
        for (k, v) in hb.hetoutputs(d)
            k in hb.hetoutputs.outputs && (d[k] = v)
        end
    end
    return d
end

"""
    mpc_statistics(block, ss; transfer, amount, consumption="c", horizons=(1, 2, 4), groups=Dict())

Finite-transfer marginal propensities to consume at the steady state of a
heterogeneous-agent block.

  * `transfer`    name of the household variable that receives the transfer. It must add
                  to cash on hand one for one (a lump-sum transfer; for a tax, pass a
                  negative `amount`). It can be a block input or an individual-level
                  hetinput (e.g. an array of transfers by productivity state).
  * `amount`      size of the one-time transfer, in the units of `transfer`.
  * `consumption` name of the individual consumption policy.
  * `horizons`    cumulative horizons in periods; 1 is the impact MPC.
  * `groups`      Dict(name => Bool array over the household state space), e.g.
                  hand-to-mouth households; group masses and mean MPCs are reported.

The household is surprised by the transfer at date 0 and expects the steady state
from then on, so continuation values and all later policies are the steady-state
ones. The cumulative MPC at horizon h is the extra expected consumption over the
first h periods divided by `amount`.

Returns a named tuple: `mpc` (Dict horizon => array over states), `stats`
(Dict horizon => (mean, median, p90, high_mpc_share, groups)), `group_mass`, and
`replay_error` (how well one backward step at the steady state reproduces the
steady-state policies; it should be at the level of the backward tolerance).
"""
function mpc_statistics(hb::HetBlock, ss; transfer, amount, consumption="c", horizons=(1, 2, 4), groups=Dict())
    amount != 0 || throw(ArgumentError("amount must be non-zero"))
    ssd = extract_ss_dict(hb, ss)
    consumption = string(consumption)
    haskey(ssd, consumption) || error("'$consumption' is not an individual-level outcome of '$(hb.name)'.")
    D = ssd["D"]
    c_ss = ssd[consumption]

    baseline    = _household_step_at_ss(hb, ssd, Dict{String,Any}())
    transferred = _household_step_at_ss(hb, ssd, Dict{String,Any}(string(transfer) => amount))

    replay_error = maximum(abs.(baseline[consumption] .- c_ss))
    for k in hb.policy
        replay_error = max(replay_error, maximum(abs.(baseline[k] .- ssd[k])))
    end

    exog     = make_exog_law_of_motion(hb, ssd)
    endog_ss = make_endog_law_of_motion(hb, ssd)
    endog_0  = make_endog_law_of_motion(hb, transferred)   # date-0 policies after the transfer
    endog_b  = make_endog_law_of_motion(hb, baseline)
    # expected value next period of X (defined on next period's states), given today's policy
    step(endog, X) = expectation(endog, expectation(exog, X))

    function cumulative(c0, endog0, h)
        continuation = zeros(size(c_ss))
        for _ in 1:(h - 1)
            continuation = c_ss .+ step(endog_ss, continuation)
        end
        return c0 .+ step(endog0, continuation)
    end

    w = vec(D)
    mpcs = Dict{Int,Any}(); stats = Dict{Int,Any}()
    group_mass = Dict{String,Float64}(string(g) => sum(D[m]) for (g, m) in groups)
    for h in horizons
        h >= 1 || throw(ArgumentError("horizons must be positive"))
        mpc = (cumulative(transferred[consumption], endog_0, h) .- cumulative(baseline[consumption], endog_b, h)) ./ amount
        all(isfinite, mpc) || error("Non-finite MPC encountered")
        v = vec(mpc)
        gm = Dict{String,Float64}()
        for (g, m) in groups
            mass = sum(D[m])
            gm[string(g)] = mass > 0 ? sum(D[m] .* mpc[m]) / mass : NaN
        end
        mpcs[h] = mpc
        stats[h] = (mean = sum(D .* mpc), median = weighted_quantile(v, w, 0.5), p90 = weighted_quantile(v, w, 0.9),
                    high_mpc_share = sum(w[v .>= 0.5]), groups = gm)
    end
    return (mpc = mpcs, stats = stats, group_mass = group_mass, amount = amount, replay_error = replay_error)
end

mpc_statistics(model::Block, ss; block=nothing, kwargs...) = mpc_statistics(_the_het_block(model, block), ss; kwargs...)
