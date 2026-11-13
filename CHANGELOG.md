# Changelog

## 0.3.0 — first public release

First public version of the Julia port of
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) (Auclert, Bardóczy,
Rognlie and Straub). It covers the whole Python toolbox: simple, solved,
heterogeneous-agent, stage and combined blocks, steady states, fake-news Jacobians,
linear and nonlinear transitions, and estimation. All example models of the Python
toolbox reproduce its numbers (see *Agreement with Python* below).

### Beyond the Python toolbox: new in this port

* **Markov transitions in stage blocks that depend on another state, or move two
  states jointly.**
  * `ExogenousMaker(name, index; by = j)`: a transition along dimension `index` that
    differs by the position in dimension `j` (array `Pi[group, from, to]`);
  * `ExogenousMaker(name, 1; joint = true)`: one transition for dimensions 1 and 2
    together (array `Pi[i, j, i', j']`).
  A shock to either kind of transition array is not implemented (an error says so).
* **Two-sided differencing of the whole household block.** `twosided = true`
  differences the backward step, the hetinputs and the hetoutputs two-sided. Python's
  flag affects the backward step only. In the two-asset household this lowers the
  relative error of the Jacobian against a brute-force one from up to 1e-3 to about
  1e-5.
* **Memory-saving stage blocks.** The Jacobian keeps the perturbed distributions for
  every input and computes expectation vectors one output at a time, so memory grows
  with the number of inputs, not the number of outcomes. The nonlinear transition uses
  two sweeps (value functions backward, policies and outcomes forward), about ten times
  less memory than storing every date's outcomes. Same numbers.
* **Jacobian of stage blocks with a `Continuous2D` stage.** Python 1.0.0 fails here
  (`ShockedPolicyLottery2D` refers to attributes that do not exist).
* **Test model `labor_choice_dag()`**: a Krusell-Smith economy with a work/home choice,
  written as a `StageBlock`, used to test stage blocks in general equilibrium.

### Beyond the Python toolbox: following Yuxuan Zhao's sequence-space notebooks

The ideas below come from checks and tools in Yuxuan Zhao's notebooks (RBC,
Krusell-Smith, one- and two-asset HANK). They are written here for arbitrary blocks.

* **Nonlinear solver with backtracking and a warm start.** `solve_impulse_nonlinear`
  accepts a step only if the largest target residual falls, and can start from the
  linear solution (`warm_start = true`). `backtrack = false` gives the Python full-step
  rule.
* **Validation tools** (`src/utilities/diagnostics.jl`): `jacobian_direct`,
  `fake_news_error`, `steady_path_error`, `distribution_distance`,
  `terminal_distribution_distance`.
* **Household statistics**: `channel_decomposition`, `mpc_statistics`, with helpers
  `weighted_quantile` and `het_blocks`.

### Python features that now work in Julia

These exist in the Python toolbox; the earlier Julia code lacked them or they did not
run.

* **Nonlinear transitions with heterogeneous agents.** `HetBlock` solves the household
  backward from the steady-state continuation value and pushes the distribution forward.
  Distribution and policy paths are available through `internals` (`"D"`, `"Dbeg"`,
  `"Dbeg_terminal"`, any individual-level outcome).
* **Stage blocks.** `StageBlock` is a full `Block`: steady state, Jacobian (fake news),
  linear and nonlinear transitions, combinable with other blocks in `create_model`.
  Stages: `ExogenousMaker` / `Exogenous`, `LogitChoice`, `Continuous1D`, `Continuous2D`.
* **Options everywhere.** Keyword arguments travel down through combined and solved
  blocks; per-block options (`options = Dict("hh" => Dict(...))`) survive inside
  `solve_steady_state`; option names may be Strings or Symbols.
* **`drawdag`** returns the model's DAG as Graphviz DOT source (and can write and
  render it). Python draws it with the `graphviz` package.
* Package structure: `Project.toml` with compat bounds, `test/runtests.jl`, numbered
  example scripts, documentation in `docs/`.
* The test-model functions accept calibration overrides as keyword arguments
  (`hank_dag(nS=7, nA=500)`); the defaults are unchanged.
* `showcase/run_showcase.jl` produces the tables and figures of `showcase/RESULTS.md`.

### Fixed (relative to earlier Julia code)

| Where | What was wrong | Effect |
|---|---|---|
| `classes/sparse_jacobians.jl` `multiply_basis` | The product rule for lead and lag operators used the opposite sign convention. Wrong whenever leads and lags are composed (failed 576 of 784 cases against dense products). | Every GE Jacobian in which a lead in one block meets a lag in another. RBC impact response of output 0.016252 instead of 0.015901. |
| `blocks/het_block.jl` defaults | Backward tolerance 1e-6, forward 1e-8, keyword overrides discarded. | Now the Python defaults (1e-8, 1e-10) and overridable. |
| `blocks/support/steady_state.jl` bounds | With `(lb, x0, ub)` unknowns the solver could stop outside the box and the steady state was re-evaluated there. | The solution is mapped back to the in-bounds point actually solved, with a warning. Bounds are matched to unknowns by name. |
| `utilities/discretize.jl` `nonlinspace` | Off by one: last grid point was infinite. | Last point equals `amax`. |
| `utilities/optimized_routines.jl` `fast_aggregate` | Contracted over the wrong dimensions. | Python semantics: `Z[t] = sum(X[t,...] .* Y[t,...])`. |
| `utilities/optimized_routines.jl` `setmin` | Silently treated arrays with more than two dimensions as matrices. | Works along the last dimension of arrays of any dimension. |
| `blocks/support/simple_displacement.jl` `apply` | Called `log` on a whole path instead of element by element. | `apply(x, log)` in a simple block now works in nonlinear transitions. |
| `utilities/function.jl` `output_list` | Output names were found with a regular expression. | Read from the parsed function: nested parentheses, multi-line returns, docstrings and comments containing `return` are handled. |
| `impulse_linear`, `solve_impulse_linear` | Argument counts did not match between methods; results held only shocks and unknowns; targets were stacked in hash order. | Rewritten after the Python implementation; existing block Jacobians are reused. |
| `td_nonlinear` | `tol`, `maxit`, `verbose` were passed where no block looked. | Passed to the solver. |
| `blocks/stage_block.jl`, `stages.jl` | Not a `Block`; called an undefined `MarkovTransition`. | Rewritten (see *Added*). |
| `utilities/drawdag.jl` | Exported but not defined. | Implemented (see *Added*). |
| `utilities/multidim.jl` `_einsum_ijb_jb_ib` | Took array slices inside the loop. | Plain loops: same numbers, about four times faster distribution steps with a discrete choice. |
| Inherited from FRBNY SSJ.jl | `using Ranges`; `dx *x = backtrack_c`; `obtain_j`; `X.shape[1]`; `interpolate_coord` set `nx` to `nxq`; undefined `yq` in `interpolate_point`; broken `@generated` `nonconcave`; wrong key lookup in `Dict * Bijection`; `set(...)` and `x.t` in `JacobianDict.apply`; `1::Tobs` and `log.diag(...)` in `estimate.jl`; `all_covariances` not matching Python's FFT formula. | All fixed. |

Smaller fixes: `rename(block; name=...)` on parent blocks, an `UndefVarError` in
`construct_parent`, scalar × `ImpulseDict`, subsetting and unions of `ImpulseDict`s
with mismatched element types. Six exported names that were never defined were removed
from the export list. Integer indexing into ordered sets was removed (it is gone in
OrderedCollections 2).

### Behaviour to be aware of

* **Hetinputs and hetoutputs are differentiated one-sided by default**, exactly as in
  Python, whose combined-function `diff` defaults to one-sided whatever is requested.
  The default Jacobian of a strongly nonlinear household block is therefore good to
  three or four digits, in both toolboxes. With `twosided = true` the relative error of
  the two-asset household Jacobian against a brute-force one falls from 7e-6–1e-3 to
  2e-7–1e-5, at twice the cost (see the guide).
* **Stage-block differencing steps follow Python**: 1e-4 for stage functions and
  hetinputs, 1e-5 for hetoutputs (`h_hetoutputs`). Python ignores a change in the
  taste-shock scale when it differentiates a `LogitChoice`; so does this port. Do not
  use the taste-shock scale as a shocked input.
* **Convergence of the distribution.** The forward iteration stops when no element of
  the distribution changes by more than `forward_tol` in one step. With a slow part
  (rare transitions) this says little about the distance from the limit: tighten
  `forward_tol` (for example to 1e-14) and check aggregates, not only the change per
  step.
* `solve_steady_state` warns when the largest target residual at the returned point
  exceeds 1e-5.
* `ImpulseDict` is always `ImpulseDict{String,Any}`; `pack` uses alphabetical order
  unless `force_order` is given.

### Agreement with Python

Largest absolute difference from Python `sequence-jacobian` 1.0.0 on its own example
models:

| Check | RBC | Krusell-Smith | One-asset HANK | Two-asset HANK |
|---|---|---|---|---|
| Steady state | 1e-15 | 2e-10 | 2e-10 | 8e-9 |
| GE Jacobian responses | 5e-16 | 2e-12 | 1e-12 | 4e-12 (10/16/4 grid) |
| `solve_impulse_linear` | 6e-17 | 2e-12 | 2e-13 | not compared |
| Household block, nonlinear | n/a | 3e-10 | through GE | through GE |
| GE nonlinear | 2e-15 | 1.5e-10 | 8e-11 | 6e-10 (10/16/4 grid) |
| GE nonlinear, larger shock | 1e-15 (10x) | 1e-9 (5x) | 7e-11 (2x) | not compared |

Also compared with Python: the two-type Krusell-Smith model with remapped household
blocks (Jacobian responses 2e-11, nonlinear 3e-10); an RBC model with labor solved
inside a `SolvedBlock` (Jacobian 1e-16, linear 3e-12, nonlinear 1e-7 because of the
nested tolerance); a stage block with a work/no-work choice (steady state 3e-9,
Jacobian 1e-9, nonlinear household transition 1e-10). Against the Krusell-Smith
notebook of Yuxuan Zhao (its calibration and grids): steady-state capital agrees to a
relative 1.4e-10.

The stage-block reference numbers with a `Continuous2D` stage were produced with the
Python bug noted above corrected.
