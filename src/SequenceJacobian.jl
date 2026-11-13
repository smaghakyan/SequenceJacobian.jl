# SequenceJacobian.jl
#
# A Julia port of the Python toolbox `sequence-jacobian`
# (https://github.com/shade-econ/sequence-jacobian) by Adrien Auclert, Bence Bardóczy,
# Matthew Rognlie and Ludwig Straub. The algorithms, the block structure and the
# example models are theirs:
#
#   Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021). "Using the
#   Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models."
#   Econometrica 89(5): 2375-2408.
#
# Parts of the code started from the Julia implementation SSJ.jl of the Federal
# Reserve Bank of New York DSGE team (https://github.com/FRBNY-DSGE/SSJ.jl).
# See README.md, NOTICE.md and CHANGELOG.md.

module SequenceJacobian


const SSJ_VERSION = "0.3.0"

using LinearAlgebra
using OrderedCollections
using NLsolve
using Roots
using FFTW
using Distributions: Normal, cdf

# ===== Utilities =====
include("utilities/bijection.jl")
include("utilities/ordered_set.jl")
include("utilities/discretize.jl")
include("utilities/interpolate.jl")
include("utilities/misc.jl")
include("utilities/optimized_routines.jl")
include("utilities/multidim.jl")
include("utilities/solvers.jl")
include("utilities/graph.jl")
include("utilities/function.jl")
include("utilities/differentiate.jl")
include("utilities/drawdag.jl")

# ===== Classes =====
include("classes/sparse_jacobians.jl")
include("classes/result_dict.jl")
include("classes/steady_state_dict.jl")
include("classes/impulse_dict.jl")
include("classes/jacobian_dict.jl")

# ===== Block support =====
include("blocks/support/het_compiled.jl")
include("blocks/support/het_support.jl")
include("blocks/support/steady_state.jl")
include("blocks/support/simple_displacement.jl")
include("blocks/support/stages.jl")

# ===== Blocks =====
include("blocks/block.jl")
include("blocks/parent_block.jl")
include("blocks/auxiliary_blocks/jacobiandict_block.jl")
include("blocks/simple_block.jl")
include("blocks/het_block.jl")
include("blocks/combined_block.jl")
include("blocks/solved_block.jl")
include("blocks/stage_block.jl")

# ===== Hetblocks =====
include("hetblocks/hh_sim.jl")
include("hetblocks/hh_labor.jl")
include("hetblocks/hh_twoasset.jl")

# ===== Estimation =====
include("estimate.jl")
include("utilities/convenience.jl")
include("utilities/diagnostics.jl")

# ===== Examples =====
include("examples/Examples.jl")
include("examples/TwoAssetHANK.jl")
include("examples/LaborChoice.jl")

# ===== Exports =====
export
    # Grids / Discretize
    asset_grid, agrid, agrid_old, nonlinspace,
    stationary, markov_tauchen, markov_rouwenhorst,
    disc_mean, disc_variance, disc_std, disc_cov, disc_corr,
    # Interpolate
    interpolate_y, interpolate_coord, apply_coord,
    interpolate_coord_robust, interpolate_point,
    # Misc
    make_tuple, demean, factor, factored_solve,
    unprime, uncapitalize, list_diff, dict_diff,
    smart_set, smart_zip, smart_zeros,
    logit, logsum, logit_choice, nonconcave, numeric_primitive,
    # Optimized
    setmin, within_tolerance, fast_aggregate,
    # Multidim
    multiply_ith_dimension, outer, batch_multiply_ith_dimension,
    # Solvers
    newton_solver, broyden_solver, brentq,
    # Differentiate
    numerical_diff, numerical_diff_symmetric,
    # Graph
    DAG, find_intermediate_inputs,
    # Function
    ExtendedFunction, CombinedExtendedFunction,
    input_list, output_list, input_defaults,
    # Drawdag
    drawdag,
    # Bijection
    Bijection, make_bijection,
    # Sparse
    IdentityMatrix, SimpleSparse, matrix, make_matrix,
    # Classes
    SteadyStateDict, ImpulseDict, JacobianDict, FactoredJacobianDict,
    jacobian_identity, addinputs, remap, pack, unpack, compose, apply,
    # Displacement
    IgnoreInt, IgnoreFloat64, IgnoreArray, Displace, AccumulatedDerivative,
    ignore, apply_function,
    # Stages
    Stage, Continuous1D, Continuous2D, Exogenous, ExogenousMaker, LogitChoice,
    DiscreteChoice, backward_step, backward_step_shock, precompute, make_stage, add_hetoutputs,
    # Transitions
    PolicyLottery1D, PolicyLottery2D, Markov,
    CombinedTransition, lottery_1d, lottery_2d,
    forward, expectation, forward_shockable, expectation_shockable,
    forward_shock, expectation_shock,
    # het_compiled
    forward_policy_1d, expectation_policy_1d, forward_policy_shock_1d,
    forward_policy_2d, expectation_policy_2d, forward_policy_shock_2d,
    # Blocks
    Block, ParentBlock,
    SimpleBlock, simple,
    HetBlock, het,
    StageBlock,
    CombinedBlock, combine, create_model,
    SolvedBlock, solved,
    # Block methods
    steady_state, impulse_nonlinear, impulse_linear, jacobian,
    # Python-identical public API
    simple, het, solved, combine, create_model,
    ImpulseDict, JacobianDict, SteadyStateDict,
    # Estimation
    all_covariances, log_likelihood, log_likelihood_formula,
    build_full_covariance_matrix,
    # Convenience wrappers (matching Python sj.block.* API)
    get_shock_path, td_nonlinear, irf_pct, build_MA_representation,
    solve_steady_state, solve_impulse_nonlinear, solve_impulse_linear,
    solve_jacobian, partial_jacobians,
    add_hetinputs, add_hetoutputs, remap, rename,
    # Validation and household statistics (diagnostics.jl)
    jacobian_direct, fake_news_error, steady_path_error,
    distribution_distance, terminal_distribution_distance,
    channel_decomposition, mpc_statistics, weighted_quantile, het_blocks,
    # SS support
    provide_solver_default, compute_target_values,
    # Hetblocks
    hh_sim, hh_labor, hh_twoasset, hh_extended,
    example_calibration, make_grids, income,
    # Examples
    rbc_dag, hank_dag, ks_dag, ks_remapped_dag, two_asset_dag, labor_choice_dag,
    # Version
    SSJ_VERSION

end # module
