using LinearAlgebra


abstract type Transition end

# Construct a 1D policy lottery given policy 'a' and grid 'a_grid'.
function lottery_1d(a, a_grid; monotonic=false)
    if !monotonic
        return PolicyLottery1D(interpolate_coord_robust(a_grid, a)..., a_grid)
    else
        return PolicyLottery1D(interpolate_coord(a_grid, a)..., a_grid)
    end
end

abstract type AbstractPolicyLottery1D <: Transition end

# Represents a 1D transition operator using linear interpolation.
struct PolicyLottery1D <: AbstractPolicyLottery1D
    i
    _pi
    grid
    flatshape
    shape
    endog_shape

    # flatten the non-policy dimensions into one for the method to accept
    function PolicyLottery1D(i, _pi, grid)
        flat_i = reshape(i, (:, size(grid)...))
        flat_pi = reshape(_pi, size(flat_i))
        shape = size(i)
        endog_shape = (size(i)[end])
        return new(flat_i, flat_pi, grid, size(flat_i), shape, endog_shape)
    end
end

# Propagate a distribution 'D' forward using the policy lottery interpolation.
function forward(plt::AbstractPolicyLottery1D, D)
    reshaped_D = reshape(D, plt.flatshape)
    result = forward_policy_1d(reshaped_D, plt.i, plt._pi)
    return reshape(result, plt.shape)
end

# Compute the expected value of 'X' under the policy lottery transition.
function expectation(plt::AbstractPolicyLottery1D, X)
    reshaped_X = reshape(X, plt.flatshape)
    result = expectation_policy_1d(reshaped_X, plt.i, plt._pi)
    return reshape(result, plt.shape)
end

# Create a shockable version of the 1D policy transition that can be used for first-order perturbation.
function forward_shockable(plt::AbstractPolicyLottery1D, Dss)
    reshaped_i = reshape(plt.i, plt.shape)
    reshaped_pi = reshape(plt._pi, plt.shape)
    return ForwardShockablePolicyLottery1D(reshaped_i, reshaped_pi, plt.grid, Dss)
end

struct ForwardShockablePolicyLottery1D <: AbstractPolicyLottery1D 
    i
    _pi
    grid
    flatshape
    shape
    endog_shape
    Dss
    space

    function ForwardShockablePolicyLottery1D(i, _pi, grid, Dss)
        flat_i = reshape(i, (:, size(grid)...))
        flat_pi = reshape(_pi, size(flat_i))
        flatshape=size(flat_i)
        shape = size(i)
        endog_shape = (size(i)[end])
        flat_Dss = reshape(Dss, flatshape)
        space = grid[i .+ 1] - grid[i]

        return new(flat_i, flat_pi, grid, flatshape, shape, endog_shape, flat_Dss, space)
    end
end

# Given a perturbation 'da' to the endogenous variable, return the resulting distributional shock.
function forward_shock(fplt::ForwardShockablePolicyLottery1D, da)
    pi_shock = -reshape(da, fplt.flatshape) ./ fplt.space
    return reshape(forward_policy_shock_1d(fplt.Dss, fplt.i, pi_shock), size(fplt.i))
end

# Construct a 2D policy lottery for two endogenous policy vars 'a' and 'b'
function lottery_2d(a, b, a_grid, b_grid; monotonic=false)
    if !monotonic
        return PolicyLottery2D(interpolate_coord_robust(a_grid, a)...,
                               interpolate_coord_robust(b_grid, b)..., a_grid, b_grid)
    else
        # we have no  monotonic 2D examples, so this should not be called
        return PolicyLottery2D(interpolate_coord(a_grid, a)...,
                               interpolate_coord(b_grid, b)..., a_grid, b_grid)
    end
end

abstract type AbstractPolicyLottery2D <: Transition end

# Represents a 2D bilinear policy lottery over two endogenous dimensions.
struct PolicyLottery2D <: AbstractPolicyLottery2D
    i1
    i2
    pi1
    pi2
    grid1
    grid2
    flatshape
    shape
    endog_shape

    function PolicyLottery2D(i1, pi1, i2, pi2, grid1, grid2)
        flat_i1 = reshape(i1, (:, size(grid1)..., size(grid2)...))
        flatshape = size(flat_i1)
        flat_i2 = reshape(i2, flatshape)
        flat_pi1 = reshape(pi1, flatshape)
        flat_pi2 = reshape(pi2, flatshape)
        shape = size(i1)
        endog_shape = (size(i1)[end-1], size(i1)[end])

        return new(flat_i1, flat_i2, flat_pi1, flat_pi2, grid1, grid2, flatshape, shape, endog_shape)
    end
end

# Propagate a distribution 'D' forward using the policy lottery interpolation.
function forward(plt::AbstractPolicyLottery2D, D)
    reshaped_D = reshape(D, plt.flatshape)
    result = forward_policy_2d(reshaped_D, plt.i1, plt.i2, plt.pi1, plt.pi2)
    return reshape(result, plt.shape)
end

# Compute the expected value of 'X' under the policy lottery transition.
function expectation(plt::AbstractPolicyLottery2D, X)
    reshaped_X = reshape(X, plt.flatshape)
    result = expectation_policy_2d(reshaped_X, plt.i1, plt.i2, plt.pi1, plt.pi2)
    return reshape(result, plt.shape)
end

# Create a shockable version of the 1D policy transition that can be used for first-order perturbation.
function forward_shockable(plt::AbstractPolicyLottery2D, Dss)
    reshaped_i1 = reshape(plt.i1, plt.shape)
    reshaped_pi1 = reshape(plt.pi1, plt.shape)
    reshaped_i2 = reshape(plt.i2, plt.shape)
    reshaped_pi2 = reshape(plt.pi2, plt.shape)
    return ForwardShockablePolicyLottery2D(reshaped_i1, reshaped_pi1, reshaped_i2, reshaped_pi2, plt.grid1, plt.grid2, Dss)
end

struct ForwardShockablePolicyLottery2D <: AbstractPolicyLottery2D
    i1
    i2
    pi1
    pi2
    grid1
    grid2
    flatshape
    shape
    endog_shape
    Dss
    space1
    space2

    function ForwardShockablePolicyLottery2D(i1, pi1, i2, pi2, grid1, grid2, Dss)
        flat_i1 = reshape(i1, (:, size(grid1)..., size(grid2)...))
        flatshape = size(flat_i1)
        flat_i2 = reshape(i2, flatshape)
        flat_pi1 = reshape(pi1, flatshape)
        flat_pi2 = reshape(pi2, flatshape)
        shape = size(i1)
        endog_shape = (size(i1)[end-1], size(i1)[end])

        flat_Dss = reshape(Dss, flatshape)
        sp1 = grid1[i1 .+ 1] .- grid1[i1]
        sp2 = grid2[i2 .+ 1] .- grid2[i2]
       
       return new(flat_i1, flat_i2, flat_pi1, flat_pi2, grid1, grid2, flatshape, shape, endog_shape, flat_Dss, sp1, sp2)

    end
end

# Given a perturbation 'da' to the endogenous variable, return the resulting distributional shock.
function forward_shock(fplt::ForwardShockablePolicyLottery2D, da)
    da1, da2 = da
    pi_shock1 = -reshape(da1, fplt.flatshape) ./ fplt.space1
    pi_shock2 = -reshape(da2, fplt.flatshape) ./ fplt.space2
    result = forward_policy_shock_2d(fplt.Dss, fplt.i1, fplt.i2, fplt.pi1, fplt.pi2, pi_shock1, pi_shock2)
    return reshape(result, fplt.shape)
end

abstract type AbstractMarkov <: Transition end

struct Markov <: AbstractMarkov
    Pi
    Pi_T
    i

    function Markov(Pi, i)
        Pi_T = transpose(Pi)
        if Pi_T isa Array
            Pi_T = copy(Pi_T)
        end
        return new(Pi, Pi_T, i)
    end
end

# Propagate distribution 'D' forward using Markov transition matrix.
function forward(mkv::AbstractMarkov, D)
    return multiply_ith_dimension(mkv.Pi_T, mkv.i, D)
end

# Compute expected value of 'X' under the Markov transition.
function expectation(mkv::AbstractMarkov, X)
    return multiply_ith_dimension(mkv.Pi, mkv.i, X)
end


function forward_shockable(mkv::AbstractMarkov, Dss)
    return ForwardShockableMarkov(mkv.Pi, mkv.i, Dss)
end

function expectation_shockable(mkv::AbstractMarkov, Xss)
    return ExpectationShockableMarkov(mkv.Pi, mkv.i, Xss)
end

function stationary(mkv::AbstractMarkov, pi_seed; tol=1e-11, maxit=10000)
    return stationary(mkv.Pi, pi_seed; tol=tol, maxit=maxit)
end

struct ForwardShockableMarkov <: AbstractMarkov
    Pi
    Pi_T
    i
    Dss
    function ForwardShockableMarkov(Pi, i, Dss)
        Pi_T = transpose(Pi)
        if Pi_T isa Array
            Pi_T = copy(Pi_T)
        end
        new(Pi, Pi_T, i, Dss)
    end
end

# Apply a vector of transition shocks to a steady-state distribution.
function forward_shock(fsm::ForwardShockableMarkov, dPi)
    dPi_T = Matrix(transpose(dPi))
    return multiply_ith_dimension(dPi_T, fsm.i, fsm.Dss)
end

struct ExpectationShockableMarkov <: AbstractMarkov
    Pi
    Pi_T
    i
    Xss
    function ExpectationShockableMarkov(Pi, i, Xss)
        Pi_T = transpose(Pi)
        if Pi_T isa Array
            Pi_T = copy(Pi_T)
        end
        new(Pi, Pi_T, i, Xss)
    end
end

# Propagate perturbations in expectations backward through a chain of transitions.
function expectation_shock(esm::ExpectationShockableMarkov, dPi)
    return multiply_ith_dimension(dPi, esm.i, esm.Xss)
end

abstract type AbstractCombinedTransition <: Transition end

# Composes multiple transition stages (e.g. Markov + policy)
struct CombinedTransition <: AbstractCombinedTransition
    stages
end

# Applies all transitions in 'ct.stages' sequentially to distribution 'D'.
function forward(ct::AbstractCombinedTransition, D)
    for stage ∈ ct.stages
        D = forward(stage, D)
    end
    return D
end

# Applies all transitions in reverse to compute expectations.
function expectation(ct::AbstractCombinedTransition, X)
    for stage ∈ reverse(ct.stages)
        X = expectation(stage, X)
    end
    return X
end

function forward_shockable(ct::AbstractCombinedTransition, Dss)
    shockable_stages = []
    for stage in ct.stages
        push!(shockable_stages, forward_shockable(stage, Dss))
        Dss = forward(stage, Dss)
    end
    return ForwardShockableCombinedTransition(shockable_stages)
end

function expectation_shockable(ct::AbstractCombinedTransition, Xss)
    shockable_stages = []
    for stage in reverse(ct.stages)
        push!(shockable_stages, expectation_shockable(stage, Xss))
        Xss = expectation(stage, Xss)
    end
    return ExpectationShockableCombinedTransition(reverse(shockable_stages))
end

function Base.getindex(ct::AbstractCombinedTransition, i::Int)
    return ct.stages[i]
end

const Shock = Any
const ListTupleShocks = Union{Vector{Shock}, NTuple{N,Shock}} where N

struct ForwardShockableCombinedTransition <: AbstractCombinedTransition
    stages
    Dss

    function ForwardShockableCombinedTransition(stages)
        Dss = stages[1].Dss
        new(stages, Dss)
    end
end

# Apply a vector of transition shocks to a steady-state distribution.
function forward_shock(fct::ForwardShockableCombinedTransition, shocks)
    if isnothing(shocks)
        return nothing
    end

    dD = nothing

    for (stage, shock) in zip(fct.stages, shocks)
        if !isnothing(shock)
            dD_shock = forward_shock(stage, shock)
        else
            dD_shock = nothing
        end

        if !isnothing(dD)
            dD = forward(stage, dD)
            if !isnothing(shock)
                dD += dD_shock
            end
        else
            dD = dD_shock
        end
    end
    return dD
end

struct ExpectationShockableCombinedTransition <: AbstractCombinedTransition
    stages
    Xss

    function ExpectationShockableCombinedTransition(stages)
        Xss = stages[end].Xss
        new(stages, Xss)
    end
end

# Propagate perturbations in expectations backward through a chain of transitions.
function expectation_shock(ect::ExpectationShockableCombinedTransition, shocks)
    dX = nothing
    
    for (stage,shock) ∈ zip(reverse(ect.stages), reverse(shocks))
        if !isnothing(shock)
            dX_shock = expectation_shock(stage, shock)
        else
            dX_shock = nothing
        end
        if !isnothing(dX)
            dX = expectation(stage, dX)
            if !isnothing(shock)
                dX += dX_shock
            end
        else
            dX = dX_shock
        end
    end
    return dX
end

const ForwardShockableTransition = Union{ForwardShockablePolicyLottery1D, ForwardShockablePolicyLottery2D, ForwardShockableMarkov, ForwardShockableCombinedTransition}

const ExpectationShockableTransition = Union{ExpectationShockableMarkov, ExpectationShockableCombinedTransition}
