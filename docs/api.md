# API reference

Every exported name of `SequenceJacobian`, grouped by purpose. Signatures show the
defaults. "Python:" gives the corresponding name in `sequence-jacobian` where it
differs. For explanations and examples see [guide.md](guide.md).

Contents: [Blocks](#blocks) · [Block methods](#block-methods) ·
[Stages](#stages) · [Result containers](#result-containers) ·
[Jacobian internals](#jacobian-internals) · [Laws of motion](#laws-of-motion) ·
[Grids and Markov chains](#grids-and-markov-chains) ·
[Interpolation](#interpolation) · [Array utilities](#array-utilities) ·
[Solvers](#solvers) · [Functions and graphs](#functions-and-graphs) ·
[Estimation](#estimation) · [Convenience wrappers](#convenience-wrappers) ·
[Validation and household statistics](#validation-and-household-statistics) ·
[Household blocks](#household-blocks) · [Test models](#test-models) ·
[Simple-block internals](#simple-block-internals)

---

## Blocks

| Name | Signature | Description |
|---|---|---|
| `Block` | abstract type | Supertype of all blocks. Every block has `name`, `inputs`, `outputs` and option dictionaries. |
| `ParentBlock` | abstract type | Blocks that contain other blocks (`CombinedBlock`, `SolvedBlock`). |
| `simple` | `simple(f)` | `SimpleBlock` from a function of aggregate variables. Inputs: the keyword arguments of `f`; outputs: the names of its returned NamedTuple. |
| `SimpleBlock` | `SimpleBlock(f::Function)` | Constructor behind `simple`. |
| `het` | `het(backward_fun; exogenous, policy, backward, backward_init=nothing, hetinputs=nothing, hetoutputs=nothing)` | `HetBlock` from a backward-iteration function. See [guide §4](guide.md#4-heterogeneous-agent-blocks). |
| `HetBlock` | `HetBlock(backward_fun, exogenous, policy, backward, backward_init=nothing, hetinputs=nothing, hetoutputs=nothing)` | Constructor behind `het`. Fields used often: `name`, `inputs`, `outputs`, `internals`, `policy`, `exogenous`, `backward`, `steady_state_options`, `jacobian_options`. |
| `StageBlock` | `StageBlock(stages; backward_init=nothing, hetinputs=nothing, name=nothing)` | Heterogeneous-agent block whose period is a list of stages. See [guide §5](guide.md#5-stage-blocks). Field `stages`. |
| `add_hetinputs` | `add_hetinputs(block, functions)` | New block with hetinput functions added (`HetBlock` or `StageBlock`). The functions' outputs feed the backward step; their inputs become block inputs. |
| `add_hetoutputs` | `add_hetoutputs(block, functions)` / `add_hetoutputs(stage, functions)` | New `HetBlock` (or stage) with hetoutput functions added. Their individual outputs are aggregated into upper-case block outputs. |
| `solved` | `solved(block, unknowns, targets; name=block.name*"_solved", solver=nothing, solver_kwargs=nothing)` | `SolvedBlock` that solves `targets` for `unknowns` internally. `unknowns` as in `solve_steady_state`. |
| `SolvedBlock` | `SolvedBlock(block, name, unknowns, targets; solver=nothing, solver_kwargs=Dict())` | Constructor behind `solved`. |
| `create_model`, `combine` | `create_model(blocks; name="")` | `CombinedBlock` from a list of blocks (and `JacobianDict`s, which are wrapped in a block), sorted topologically. |
| `CombinedBlock` | `CombinedBlock(blocks; name="", model_alias=false)` | Constructor behind `create_model`. Fields: `blocks` (sorted), `inputs`, `outputs`, `adj`, `revadj`. |
| `remap` | `remap(block, map::Dict)` | Copy of `block` with variables renamed by `map` (inputs and outputs). |
| `rename` | `rename(block; name=nothing, suffix=nothing)` | Copy with a new name, or with `suffix` appended (for parent blocks the suffix is also applied to the blocks inside). |

---

## Block methods

All methods work on any block, including whole models. `options` is
`Dict(block name => Dict(option => value))`; other keyword arguments are options passed
to every block that has them (see [guide §13](guide.md#13-options-and-accuracy)).

| Name | Signature | Returns |
|---|---|---|
| `steady_state` | `steady_state(block, calibration; dissolve=[], options=Dict(), kwargs...)` | `SteadyStateDict` with every input and output; heterogeneous-agent blocks add `internals`. |
| `solve_steady_state` | `solve_steady_state(block, calibration, unknowns, targets; dissolve=[], options=Dict(), solver="", solver_kwargs=Dict(), ttol=1e-12, verbose=false, constrained_method="linear_continuation", constrained_kwargs=Dict())` | `SteadyStateDict` at which `targets` hold. `unknowns`: `Dict(name => start)`, `=> (lower, upper)` or `=> (lower, start, upper)`. `targets`: `Dict(name => value)`; a String value names another variable. Warns if the largest residual exceeds 1e-5. |
| `jacobian` | `jacobian(block, ss, inputs, outputs=nothing; T=nothing, Js=Dict(), options=Dict(), kwargs...)` | `JacobianDict`: partial-equilibrium Jacobians of `outputs` with respect to `inputs`. HetBlock options: `h=1e-4`, `twosided=false`. StageBlock: also `h_hetoutputs=1e-5`. |
| `partial_jacobians` | `partial_jacobians(block, ss, inputs=nothing, outputs=nothing; T=nothing, Js=Dict(), options=Dict())` | `Dict(block name => JacobianDict)` for every block inside `block`. Pass back as `Js`. |
| `solve_jacobian` | `solve_jacobian(block, ss, unknowns, targets, inputs, outputs=nothing; T=300, Js=Dict(), options=Dict(), H_U_factored=nothing)` | General-equilibrium `JacobianDict` `G` of all outputs and unknowns with respect to `inputs`. |
| `impulse_linear` | `impulse_linear(block, ss, inputs, outputs=nothing; Js, options)` | `ImpulseDict` of first-order responses to the input paths (`Dict` or `ImpulseDict`), holding everything else at the steady state. |
| `solve_impulse_linear` | `solve_impulse_linear(block, ss, unknowns, targets, inputs, outputs=nothing; Js=Dict(), options=Dict(), H_U_factored=nothing)` | `ImpulseDict` of general-equilibrium first-order responses (shocks, unknowns and all scalar outputs). |
| `impulse_nonlinear` | `impulse_nonlinear(block, ss, inputs; outputs=nothing, internals=Dict(), Js=Dict(), options=Dict(), ss_initial=nothing)` | `ImpulseDict` of nonlinear responses to the input paths, unknowns held at the given paths. `internals`: `Dict(block name => names)` or a list of block names. |
| `solve_impulse_nonlinear` | `solve_impulse_nonlinear(block, ss, unknowns, targets, inputs; outputs=nothing, internals=Dict(), Js=Dict(), options=Dict(), H_U_factored=nothing, ss_initial=nothing, tol=1e-8, maxit=30, verbose=true, backtrack=true, max_backtracks=25, warm_start=false)` | `ImpulseDict` of the nonlinear perfect-foresight transition. Errors if it does not converge. |

---

## Stages

Building blocks of a `StageBlock`. Every stage has `name`, `inputs`, `report` (its
outcomes), `backward_outputs` and optional `hetoutputs`.

| Name | Signature | Description |
|---|---|---|
| `Stage` | abstract type | Supertype of the stages. |
| `ExogenousMaker` | `ExogenousMaker(markov_name, index; name="exog_<markov_name>", hetoutputs=nothing, by=0, joint=false)` | Exogenous Markov transition along dimension `index`; turned into an `Exogenous` stage with the backward variables of the next stage. `by=j`: matrix differs by position in dimension `j` (`Pi[group, from, to]`). `joint=true`, `index=1`: one transition `Pi[i, j, i', j']` for dimensions 1 and 2. |
| `Exogenous` | `Exogenous(markov_name, index, name, backward; hetoutputs=nothing, by=0, joint=false)` | The stage itself, if the backward variables are known. |
| `LogitChoice` | `LogitChoice(value, backward, index, taste_shock_scale; f=nothing, name=nothing, hetoutputs=nothing)` | Discrete choice of dimension `index` with extreme-value taste shocks. `f`: flow utility, returning one array `flow[choice, state...]` (`-Inf` for unavailable choices). |
| `Continuous1D` | `Continuous1D(backward, policy, f; name=nothing, hetoutputs=nothing)` | Continuous choice of one endogenous state (last dimension), by the stage function `f`. The policy's grid is the input `<policy>_grid`. |
| `Continuous2D` | `Continuous2D(backward, policies, f; name=nothing, hetoutputs=nothing)` | The same with two endogenous states (last two dimensions). |
| `DiscreteChoice` | `DiscreteChoice(P, index)` | Law of motion of a discrete choice: `P[d, state...]` is the probability of choice `d`. |
| `make_stage` | `make_stage(maker, backward)` | Turn a stage maker into a stage. |
| `backward_step` | `backward_step(stage, inputs; lawofmotion=false)` | One backward step of a stage; with `lawofmotion=true` also the law of motion. |
| `backward_step_shock` | `backward_step_shock(stage, ss, shocks, pre)` | First-order backward step (used by the fake-news algorithm). |
| `precompute` | `precompute(stage, ss, lom; h=1e-4, twosided=false)` | Steady-state objects reused by `backward_step_shock`. |

---

## Result containers

| Name | Signature | Description |
|---|---|---|
| `SteadyStateDict` | `SteadyStateDict(data::AbstractDict; internals=nothing)` | Steady-state values. `ss["x"]`, `ss[["x", "y"]]`, `ss.toplevel`, `ss.internals[block name]`, `merge!`, `copy`, `keys`, `haskey`. |
| `ImpulseDict` | `ImpulseDict(data::AbstractDict; internals=nothing, T=nothing)` | Paths of deviations. `irf["x"]`, `irf[["x", "y"]]`, `get(irf, names)`, `irf.T`, `irf.internals`; `+ - * /` with `ImpulseDict`s, `SteadyStateDict`s and numbers; `∪`. |
| `JacobianDict` | `JacobianDict(nesteddict; outputs=nothing, inputs=nothing, name=nothing, T=nothing)` | Jacobians `J[output][input]`. `J[outputs, inputs]` selects. `J.outputs`, `J.inputs`, `J.T`. |
| `FactoredJacobianDict` | `FactoredJacobianDict(J::JacobianDict; T=nothing)` | LU factorization of a square `JacobianDict` (e.g. `H_U`). `apply(F, dH)` solves `H_U dU = -dH`; `F * J` composes. |
| `jacobian_identity` | `jacobian_identity(names)` | Identity `JacobianDict` on `names`. Python: `JacobianDict.identity`. |
| `addinputs` | `addinputs(J)` | Add identity entries for inputs that are not outputs. |
| `compose` | `compose(J, K)` | Chain rule `J ∘ K`. Also `J * K`. |
| `apply` | `apply(J, shocks)` | Responses `J * dx` for a `Dict`/`ImpulseDict` of paths; returns the shocks and the responses. Also `J * shocks`. (For simple blocks, `apply(x, f)` applies a scalar function to a path; see [simple-block internals](#simple-block-internals).) |
| `pack` | `pack(J::JacobianDict; T=nothing)`, `pack(irf::ImpulseDict; force_order=nothing)` | One big matrix (outputs × inputs, blocks of `T × T`), or one stacked vector of paths. |
| `unpack` | `unpack(M, outputs, inputs, T)`, `unpack(v, outputs, T)` | Inverse of `pack`. |
| `Bijection` | `Bijection(map::Dict)` | Renaming map; names not in the map map to themselves. `inv(b)`, composition `b1 * b2`, application to sets, dicts and results. |
| `make_bijection` | `make_bijection(map)` | Constructor behind `Bijection` (checks that the map is one-to-one). |

---

## Jacobian internals

| Name | Signature | Description |
|---|---|---|
| `IdentityMatrix` | `IdentityMatrix()` | Identity operator of any size. |
| `SimpleSparse` | `SimpleSparse(Dict((i, m) => x, ...))` | Sparse lead/lag operator: `x` times ones on diagonal `i` with the first `m` entries missing. Jacobians of simple blocks are of this type. Supports `+`, `*` with each other and with matrices and vectors. |
| `matrix` | `matrix(s, T)` | `T × T` dense matrix of a `SimpleSparse` or `IdentityMatrix`. |
| `make_matrix` | `make_matrix(A, T)` | `matrix(A, T)`, or `A` itself if it is already an array. |

---

## Laws of motion

Used inside heterogeneous-agent blocks; useful when writing household statistics.

| Name | Signature | Description |
|---|---|---|
| `lottery_1d` | `lottery_1d(a, a_grid; monotonic=false)` | `PolicyLottery1D`: each point moves to the two grid points around its policy, with weights that preserve the mean. |
| `lottery_2d` | `lottery_2d(a, b, a_grid, b_grid; monotonic=false)` | `PolicyLottery2D` for two endogenous states. |
| `PolicyLottery1D`, `PolicyLottery2D` | constructors from indices and weights | The lotteries. |
| `Markov` | `Markov(Pi, i)` | Markov transition with matrix `Pi` along dimension `i`. |
| `CombinedTransition` | `CombinedTransition([lom1, lom2, ...])` | Several laws of motion applied in turn. |
| `forward` | `forward(lom, D)` | Distribution after the transition. |
| `expectation` | `expectation(lom, X)` | Expectation, before the transition, of an array defined after it. |
| `forward_shockable`, `expectation_shockable` | `(lom, Dss)`, `(lom, Xss)` | Versions that accept first-order perturbations of the law of motion. |
| `forward_shock`, `expectation_shock` | `(lom, dshock)` | First-order change in the forward step or the expectation. |
| `forward_policy_1d`, `expectation_policy_1d`, `forward_policy_shock_1d` | `(D, x_i, x_pi)`, `(X, x_i, x_pi)`, `(Dss, x_i_ss, x_pi_shock)` | Low-level loops of the 1D lottery. |
| `forward_policy_2d`, `expectation_policy_2d`, `forward_policy_shock_2d` | the 2D analogues | Low-level loops of the 2D lottery. |

---

## Grids and Markov chains

| Name | Signature | Description |
|---|---|---|
| `agrid` | `agrid(amax, n; amin=0)` | Asset grid, evenly spaced in `log10(a + pivot)` with pivot `abs(amin) + 0.25`: dense near `amin`. Used by the Krusell-Smith and HANK examples. |
| `asset_grid` | `asset_grid(amin, amax, n)` | Double-exponential grid of the Python toolbox's `asset_grid`: dense near `amin`. Used by `make_grids`. |
| `agrid_old` | `agrid_old(amax, N, amin=0, frac=1/25)` | Grid of the older Python toolbox. |
| `nonlinspace` | `nonlinspace(amax, n, phi, amin=0)` | Grid with curvature `phi` (`phi = 1` is linear). |
| `markov_rouwenhorst` | `markov_rouwenhorst(rho, sigma, N=7)` | Rouwenhorst discretization of a log AR(1) with persistence `rho` and cross-sectional standard deviation `sigma`. Returns `(y, pi, Pi)`: levels normalized to mean 1, stationary distribution, transition matrix. |
| `markov_tauchen` | `markov_tauchen(rho, sigma, N=7, m=3; normalize=true)` | Tauchen discretization; same returns. |
| `stationary` | `stationary(Pi, pi_seed=nothing; tol=1e-11, maxit=10_000)` | Stationary distribution of a Markov matrix. |
| `disc_mean`, `disc_variance`, `disc_std` | `(x, pi)` | Moments of a discrete distribution. Python: `mean`, `variance`, `std`. |
| `disc_cov`, `disc_corr` | `(x, y, pi)` | Covariance and correlation. |

---

## Interpolation

Linear interpolation along the last dimension; arrays with up to three dimensions.

| Name | Signature | Description |
|---|---|---|
| `interpolate_y` | `interpolate_y(x, xq, y)` | Values at `xq` of the function with values `y` at increasing points `x`; `xq` must be increasing too. Extrapolates linearly. |
| `interpolate_coord` | `interpolate_coord(x, xq)` | Indices `i` and weights `pi` such that `xq = pi * x[i] + (1 - pi) * x[i+1]`. |
| `apply_coord` | `apply_coord(x_i, x_pi, y)` | Apply indices and weights from `interpolate_coord` to `y`. |
| `interpolate_coord_robust` | `interpolate_coord_robust(x, xq; check_increasing=false)` | Like `interpolate_coord` without requiring `xq` to be ordered (binary search). |
| `interpolate_point` | `interpolate_point(x, x0, x1, y0, y1)` | Linear interpolation at one point. |

---

## Array utilities

| Name | Signature | Description |
|---|---|---|
| `setmin` | `setmin(x, xmin)` | In place, `x = max(x, xmin)` along the last dimension, which must be ascending. |
| `within_tolerance` | `within_tolerance(x1, x2, tol)` | `maximum(abs.(x1 - x2)) <= tol`, stopping early. |
| `fast_aggregate` | `fast_aggregate(X, Y)` | `Z[t] = sum(X[t, ...] .* Y[t, ...])` for paths with time first. |
| `multiply_ith_dimension` | `multiply_ith_dimension(Pi, i, X)` | `Pi` times `X` along dimension `i`. |
| `batch_multiply_ith_dimension` | `batch_multiply_ith_dimension(P, i, X)` | A different matrix for each position of the other dimensions. |
| `outer` | `outer(pis)` | Outer product of a list of vectors (joint distribution of independent states). |
| `demean` | `demean(x)` | `x .- mean(x)`. |
| `factor`, `factored_solve` | `factor(X)`, `factored_solve(Z, y)` | Factorize a matrix; solve with the factorization. |
| `logit`, `logsum`, `logit_choice` | `(V, scale)` | Choice probabilities, expected maximum, and both, along dimension 1. |
| `nonconcave` | `nonconcave(Va)` | Bounds of the region where a value-function derivative is not decreasing. |
| `make_tuple` | `make_tuple(x)` | `x` as a tuple. |
| `unprime`, `uncapitalize` | `(s::String)` | Remove a trailing `_p`; lower-case the first letter. |
| `list_diff`, `dict_diff` | `(l1, l2)`, `(d1, d2)` | Elements of the first not in the second. |
| `smart_set`, `smart_zip`, `smart_zeros` | | Small helpers from the Python toolbox. |
| `numeric_primitive` | `numeric_primitive(x)` | Underlying number or array of a wrapped value. |

---

## Solvers

| Name | Signature | Description |
|---|---|---|
| `newton_solver` | `newton_solver(f, x0, y0=nothing; tol=1e-9, maxcount=100, backtrack_c=0.5, verbose=true)` | Newton's method with a numerical Jacobian and backtracking. Returns `(x, y)`. |
| `broyden_solver` | `broyden_solver(f, x0, y0=nothing; tol=1e-9, maxcount=100, backtrack_c=0.5, verbose=true)` | Broyden's method with backtracking. Returns `(x, y)`. |
| `brentq` | `brentq(f, a, b; tol=1e-12, maxiter=100)` | Brent's method on the bracket `[a, b]`. |
| `provide_solver_default` | `provide_solver_default(unknowns)` | `"brentq"` for one bracketed unknown, `"broyden_custom"` for several. |
| `compute_target_values` | `compute_target_values(targets, potential_args)` | Residuals of `targets` (a list or a `Dict(name => value or name)`). |
| `numerical_diff` | `numerical_diff(func, ssinputs_dict, shock_dict; h=1e-4, y_ss_list=nothing)` | One-sided derivative of `func` in the direction `shock_dict`. |
| `numerical_diff_symmetric` | `numerical_diff_symmetric(func, ssinputs_dict, shock_dict; h=1e-4)` | Two-sided derivative. |

---

## Functions and graphs

| Name | Signature | Description |
|---|---|---|
| `ExtendedFunction` | `ExtendedFunction(f)`, `ExtendedFunction(f, name, inputs, outputs)` | A function with named inputs and outputs. The second form is for functions without a source file. Call with a `Dict` of inputs. |
| `CombinedExtendedFunction` | `CombinedExtendedFunction(fs; name=nothing)` | Several functions evaluated in dependency order (how hetinputs and hetoutputs are stored). |
| `input_list` | `input_list(f)` | Names of the keyword arguments of `f`. |
| `output_list` | `output_list(f)` | Names in the NamedTuple(s) returned by `f`, read from its source. |
| `input_defaults` | `input_defaults(f)` | Default values of inputs (empty: block functions have none). |
| `DAG` | `DAG(blocks)` | Topologically sorted blocks with input and output maps. Errors on cycles and on outputs produced twice. |
| `find_intermediate_inputs` | `find_intermediate_inputs(blocks)` | Inputs of some block that are outputs of another. |
| `drawdag` | `drawdag(model; exogenous=String[], unknowns=String[], targets=String[], leftright=false, save=false, savepath=nothing, render=false, format="png")` | Graphviz DOT source of the model's DAG; optionally written to a file and rendered with `dot`. |

---

## Estimation

| Name | Signature | Description |
|---|---|---|
| `all_covariances` | `all_covariances(M::Array{Float64,3}, sigmas::Vector{Float64})` | `Sigma[l+1, i, j] = Cov(x_i(t), x_j(t+l))` for `l = 0..T-1`, from impulse responses `M[t, o, z]` to unit innovations and shock standard deviations `sigmas`. FFT, as in Python. |
| `log_likelihood` | `log_likelihood(Y::Matrix{Float64}, Sigma; sigma_measurement=nothing)` | Gaussian log likelihood of the data `Y[t, o]` given `Sigma`, with optional measurement-error standard deviations. |
| `build_full_covariance_matrix` | `build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)` | The `(Tobs·O) × (Tobs·O)` covariance matrix of the stacked data. |
| `log_likelihood_formula` | `log_likelihood_formula(y, V)` | `-(log det V + y' V⁻¹ y) / 2` by Cholesky. |
| `build_MA_representation` | `build_MA_representation(G, observables, shocks, T)` | `M[t, o, z]` from the first columns of `G` (i.i.d. shocks). |

---

## Convenience wrappers

| Name | Signature | Description |
|---|---|---|
| `impulse_linear` | `impulse_linear(model, ss, inputs::Dict, unknowns, targets, exog_names; T=300, Js=Dict(), outputs=nothing)` | General-equilibrium linear responses and `G` in one call: returns `(irfs::Dict, G)`. Scalars are held for `T` periods; short vectors are padded. |
| `td_nonlinear` | `td_nonlinear(model, ss, inputs::Dict, unknowns, targets, exog_names=nothing; T=300, maxit=30, tol=1e-8, verbose=true, backtrack=true, warm_start=false, internals=Dict(), Js=Dict(), H_U_factored=nothing)` | Wrapper of `solve_impulse_nonlinear` with the same input conventions. |
| `get_shock_path` | `get_shock_path(T; rho=0.9, scale=-0.0025)` | `scale .* rho .^ (0:T-1)`. |
| `irf_pct` | `irf_pct(G, var, shock, dr; ss_val=1.0)` | `100 * G[var][shock] * dr / ss_val`. |

---

## Validation and household statistics

Not in the Python toolbox; they follow checks in Yuxuan Zhao's sequence-space notebooks
(`src/utilities/diagnostics.jl`).

| Name | Signature | Returns |
|---|---|---|
| `jacobian_direct` | `jacobian_direct(block, ss, inputs, outputs=nothing; T=10, h=1e-6)` | Brute-force `JacobianDict` from `T` nonlinear transitions per input. |
| `fake_news_error` | `fake_news_error(block, ss, inputs, outputs=nothing; T=10, h=1e-6, h_direct=h, twosided=false)` | `(max_error, errors, direct, fake_news)`: largest absolute difference between `jacobian` and `jacobian_direct`, per pair, and both Jacobians. |
| `steady_path_error` | `steady_path_error(model, ss, unknowns, targets; T=20)` | `(max_error, errors, steady_state_levels)` when the steady state is fed through the nonlinear transition. |
| `distribution_distance` | `distribution_distance(model, ss, paths; block=nothing)` | `(path, terminal)`: largest deviation of the distribution from the steady state at each date, and after the last date, along the paths of the unknowns and shocks in `paths`. |
| `terminal_distribution_distance` | `terminal_distribution_distance(irf, block_name)` | Distance after the last date, from a transition run with `internals=Dict(name => ["D", "Dbeg_terminal"])`. |
| `channel_decomposition` | `channel_decomposition(model, ss, G, shocks; block=nothing, output="C", direct=String[], T=nothing, Js=Dict())` | `(contributions, input_paths, direct, indirect, total, reconstructed, closure_error)`. |
| `mpc_statistics` | `mpc_statistics(hh, ss; transfer, amount, consumption="c", horizons=(1, 2, 4), groups=Dict())` (also `(model, ss; block=...)`) | `(mpc, stats, group_mass, amount, replay_error)`; `stats[h]` has `mean`, `median`, `p90`, `high_mpc_share`, `groups`. |
| `het_blocks` | `het_blocks(block)` | All `HetBlock`s inside a model. |
| `weighted_quantile` | `weighted_quantile(values, weights, probability)` | First value whose cumulative weight reaches `probability`. |

---

## Household blocks

The households of the Python toolbox (`sequence_jacobian.hetblocks`).

| Name | Inputs | Outputs | Description |
|---|---|---|---|
| `hh_sim` | `a_grid, y, r, beta, eis, Pi` | `A, C` | Standard incomplete markets. Python: `hetblocks.hh_sim.hh`. |
| `hh_extended` | `w, rho_e, sd_e, n_e, min_a, max_a, n_a, r, beta, eis` | `A, C` | `hh_sim` with hetinputs `income` and `make_grids`. |
| `hh_labor` | `a_grid, we, T, r, beta, eis, frisch, vphi, Pi` | `A, C, N` | With endogenous labor supply. Python: `hetblocks.hh_labor.hh`. |
| `hh_twoasset` | `a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Pi` | `A, B, C, UCE, CHI` | Liquid and illiquid assets with a convex portfolio adjustment cost. Python: `hetblocks.hh_twoasset.hh`. |
| `make_grids` | `make_grids(; rho_e, sd_e, n_e, min_a, max_a, n_a)` | `e_grid, Pi, a_grid` | Hetinput for `hh_sim`. |
| `income` | `income(; w, e_grid)` | `y` | Hetinput for `hh_sim`. |
| `example_calibration` | `example_calibration()` | `Dict` | Calibration for `hh_extended`. |

---

## Test models

Each returns the models, the steady state and the general-equilibrium setup. Keyword
arguments replace entries of the default calibration, for example
`hank_dag(nS=7, nA=500)` for the grids of the Python notebooks; an unknown name is an
error. The defaults are the small grids of the Python test models.

| Name | Returns |
|---|---|
| `rbc_dag(; overrides...)` | `(model, ss, unknowns, targets, exogenous)` |
| `ks_dag(; overrides...)` | `(model_ss, ss, model, unknowns, targets, exogenous)` |
| `ks_remapped_dag(; overrides...)` | `(model_ss, ss, model, unknowns, targets, exogenous)` |
| `hank_dag(; overrides...)` | `(model_ss, ss, model, unknowns, targets, exogenous)` |
| `two_asset_dag(; overrides...)` | `(model_ss, ss, model, unknowns, targets, exogenous)` |
| `labor_choice_dag(; overrides...)` | `(model_ss, ss, model, unknowns, targets, exogenous)` |

`SSJ_VERSION` is the version string of the package.

---

## Simple-block internals

When a simple block is evaluated, its inputs are wrapped so that the same function
gives steady-state values, Jacobians and nonlinear paths. You rarely need these types
directly.

| Name | Description |
|---|---|
| `IgnoreInt`, `IgnoreFloat64`, `IgnoreArray`, `ignore(x)` | Steady-state values: `x(-1)` and `x(+1)` return `x`. |
| `Displace` | A time path with its steady state, for nonlinear transitions: `x(-1)` shifts the path. |
| `AccumulatedDerivative` | A first-order perturbation that tracks leads and lags, for Jacobians. |
| `apply` | `apply(x, f; kwargs...)`: apply a scalar function `f` to any of the above. Write `apply(x, log)` for `log(x)` in simple blocks. |
| `apply_function` | `apply_function(func, args...; kwargs...)`: apply a function date by date to several paths (steady state and nonlinear transitions only; not implemented for Jacobians). |
