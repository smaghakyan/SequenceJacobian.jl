# Guide to SequenceJacobian.jl

This guide explains how the toolbox works and how to use each part of it. It follows the
design of the Python toolbox
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) by Auclert,
Bardóczy, Rognlie and Straub, and the method of their paper (Auclert, Bardóczy, Rognlie
and Straub, 2021, *Econometrica*). Function-by-function details are in
[api.md](api.md).

Contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Conventions](#2-conventions)
3. [Simple blocks](#3-simple-blocks)
4. [Heterogeneous-agent blocks](#4-heterogeneous-agent-blocks)
5. [Stage blocks](#5-stage-blocks)
6. [Solved blocks](#6-solved-blocks)
7. [Assembling a model](#7-assembling-a-model)
8. [Steady states](#8-steady-states)
9. [Jacobians](#9-jacobians)
10. [Linear impulse responses](#10-linear-impulse-responses)
11. [Nonlinear transitions](#11-nonlinear-transitions)
12. [Results: SteadyStateDict, ImpulseDict, JacobianDict](#12-results-steadystatedict-impulsedict-jacobiandict)
13. [Options and accuracy](#13-options-and-accuracy)
14. [Estimation](#14-estimation)
15. [Validation and household statistics](#15-validation-and-household-statistics)
16. [The test models](#16-the-test-models)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. The idea in one page

A model is a set of blocks. Each block maps sequences of inputs
`{x_t}_{t=0}^{T-1}` into sequences of outputs. A simple block does this with an explicit
formula. A heterogeneous-agent block solves a household problem backward in time and
aggregates it with a distribution moved forward in time. Blocks are connected through
their names: an output of one block with the same name as an input of another feeds it.
The connections form a directed acyclic graph (DAG).

General equilibrium is a set of **targets** (market-clearing conditions, Euler
equations, Phillips curves) that must be zero at every date, solved for a set of
**unknowns** (prices, quantities) given the paths of **exogenous** shocks:

    H(U, Z) = 0.

Around the steady state, the toolbox computes the sequence-space Jacobians `H_U` and
`H_Z` by composing the Jacobians of the blocks along the DAG. Heterogeneous-agent blocks
use the fake-news algorithm, which costs about one backward iteration per input instead
of `T`. The general-equilibrium response of the unknowns is

    dU = -H_U^{-1} H_Z dZ,     G_U = -H_U^{-1} H_Z,

and every other output follows by composing again. `solve_jacobian` returns these
general-equilibrium Jacobians `G`. `solve_impulse_nonlinear` solves `H(U, Z) = 0`
exactly with a quasi-Newton method that reuses `H_U`.

---

## 2. Conventions

These follow the Python toolbox.

* **Variables are named.** Inputs and outputs are matched by name across blocks. Names
  are Strings (`"r"`, `"w"`). Each name can be the output of at most one block.
* **Block functions take keyword arguments only**, named after the variables they use:
  `function firm(; K, L, Z, alpha, delta)`.
* **Outputs are returned as a NamedTuple**: `return (r=r, w=w, Y=Y)`. The output names
  are read from the function's source (its last `return` statement and any other
  `return` with a NamedTuple). Define block functions in a file. A function created at
  the REPL or as an anonymous closure has no source file: wrap it with
  `ExtendedFunction(f, name, inputs, outputs)` or set `block.outputs` by hand.
* **Time.** Paths are vectors indexed by date, date 0 first. In simple blocks `X(-1)` is
  the lag and `X(+1)` the lead of `X`; before date 0 a lag takes the initial steady-state
  value, after date `T-1` a lead takes the terminal steady-state value.
* **Heterogeneous-agent blocks.** The backward variable of next period carries the
  suffix `_p` (`Va_p`); the function returns the current one (`Va`). Individual outcomes
  are lower case (`a`, `c`, `n`); the block reports their aggregates in upper case
  (`A`, `C`, `N`).
* **Dimensions start at 1.** In stage blocks, `ExogenousMaker("Pi", 2)` acts on the
  second dimension of the state space.
* **Steady-state values** are looked up by name: `ss["Y"]`. Responses are deviations from
  the steady state, in levels.

---

## 3. Simple blocks

A simple block is a function of aggregate variables.

```julia
function nkpc(; pi, w, Z, Y, r, mu, kappa)
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * apply(1 + pi(+1), log) / (1 + r(+1)) - apply(1 + pi, log)
    return (nkpc_res=nkpc_res,)
end
blk = simple(nkpc)          # inputs: pi, w, Z, Y, r, mu, kappa; output: nkpc_res
```

* `+`, `-`, `*`, `/` and `^` work directly on time paths.
* **Any other function** (`log`, `exp`, `sqrt`, ...) is written `apply(x, f)`: for
  example `apply(Y, log)` for `log(Y)`. The same expression then works in the steady
  state, in Jacobians and in nonlinear transitions. Python writes `np.log(Y)`; here the
  function goes through `apply`.
* Comparisons and branches (`max`, `min`, `if`) on time paths are not supported in
  simple blocks. Use a smooth formula, or put the kink in a heterogeneous-agent block.
* Jacobians of simple blocks are exact: they are found by differentiating the function
  around the steady state and stored as sparse lead/lag operators (`SimpleSparse`).
* `steady_state(blk, Dict(...))` evaluates the function at given inputs.

---

## 4. Heterogeneous-agent blocks

A `HetBlock` solves a household problem on a grid. You write one function: the
backward step, which maps next period's backward variables into today's backward
variables and policies.

```julia
function household(; Va_p, a_grid, y, r, beta, eis)
    uc_nextgrid = beta .* Va_p                        # Va_p: marginal value of assets tomorrow
    c_nextgrid = uc_nextgrid .^ (-eis)
    coh = (1 + r) .* reshape(a_grid, 1, :) .+ reshape(y, :, 1)
    a = interpolate_y(c_nextgrid .+ reshape(a_grid, 1, :), coh, reshape(a_grid, 1, :))
    setmin(a, a_grid[1])                              # borrowing constraint
    c = coh .- a
    Va = (1 + r) .* c .^ (-1 / eis)
    return (Va=Va, a=a, c=c)
end

hh = het(household; exogenous="Pi", policy="a", backward="Va", backward_init=household_init)
```

The arguments of `het`:

| Argument | Meaning |
|---|---|
| `exogenous` | Name of the Markov matrix (or list of names) of the exogenous states. `Pi[i, j]` is the probability of moving from state `i` to `j`. With several, the first acts on the first dimension, and so on. |
| `policy` | Name of the endogenous policy (`"a"`), or two names for two endogenous states (`["b", "a"]`). The grid must be an input named `<policy>_grid`. |
| `backward` | Name of the backward variable(s) (`"Va"`, or `["Vb", "Va"]`). The function takes them with the suffix `_p`. |
| `backward_init` | Function that gives starting values of the backward variables for the steady-state iteration. |
| `hetinputs`, `hetoutputs` | Optional; usually added later with `add_hetinputs`, `add_hetoutputs`. |

**State space.** Arrays are `(exogenous states..., endogenous states...)`: in the
example, `(nE, nA)`. The distribution `D` has the same shape.

**Timing within a period.** The distribution at the beginning of the period is over
(last period's exogenous state, assets). The exogenous state moves first (with `Pi`),
then households choose, then assets move along the policy (by a lottery between the two
nearest grid points). `ss.internals[hh.name]` holds `"Dbeg"` (beginning of period) and
`"D"` (after the exogenous shock).

**Hetinputs** are functions that compute individual-level inputs from aggregates before
the backward step, for example grids and income:

```julia
labor_income(; w, e_grid) = (y = w .* e_grid,)
hh = add_hetinputs(hh, [labor_income, make_grids])
```

**Hetoutputs** compute individual-level outcomes after the backward step; their
aggregates become outputs of the block:

```julia
constrained(; a, a_grid) = (htm = Float64.(a .<= a_grid[1] + 1e-10),)
hh = add_hetoutputs(hh, [constrained])         # adds the output HTM
```

**Packaged households.** `hh_sim` (standard incomplete markets), `hh_labor` (with
endogenous labor supply) and `hh_twoasset` (liquid and illiquid assets with a portfolio
adjustment cost) are the households of the Python toolbox. `hh_extended` is `hh_sim`
with the hetinputs `income` and `make_grids` already added, and `example_calibration()`
gives a calibration for it.

---

## 5. Stage blocks

A `StageBlock` is a heterogeneous-agent block whose period is a list of stages, in the
order in which they happen. It is the natural way to write discrete choices (to work, to
enrol, to default, to move) together with continuous ones.

```julia
stages = [ExogenousMaker("Pi_z", 2; name="prod"),                         # z moves (dimension 2)
          LogitChoice("V", ["Va"], 1, "taste_scale"; f=flow, name="labor"), # choose s (dimension 1)
          Continuous1D(["V", "Va"], "a", consav; name="consav")]          # choose a (dimension 3)
hh = StageBlock(stages; name="hh", backward_init=init, hetinputs=[grids, income])
```

**Stages.**

| Stage | What it does |
|---|---|
| `ExogenousMaker(markov_name, index; name, hetoutputs, by=0, joint=false)` | A Markov transition along dimension `index`, with the matrix given as an input named `markov_name`. Its backward variables are those of the next stage, so it is written as a "maker". `by = j`: a different matrix for each position in dimension `j`, as an array `Pi[group, from, to]`. `joint = true` (with `index = 1`): one transition for dimensions 1 and 2 together, `Pi[i, j, i', j']`. |
| `LogitChoice(value, backward, index, taste_shock_scale; f, name, hetoutputs)` | A discrete choice of the value of dimension `index`, with type-1 extreme-value taste shocks of scale given by the input `taste_shock_scale`. `value` is the value function (e.g. `"V"`), `backward` the other backward variables carried through the choice (e.g. `["Va"]`). `f` is the flow utility of each choice. |
| `Continuous1D(backward, policy, f; name, hetoutputs)` | A continuous choice of one endogenous state (the last dimension). `f` is the stage function, `policy` the name of the policy (its grid is the input `<policy>_grid`). |
| `Continuous2D(backward, policies, f; name, hetoutputs)` | The same with two endogenous states (the last two dimensions). |

**Writing stage functions.**

* A stage function takes the end-of-stage backward variables **under their own names**
  (`V`, `Va`, no `_p`) and returns the beginning-of-stage ones under the same names,
  together with its outcomes. In the last stage of the period, the end-of-stage values
  are next period's beginning-of-period values, so discounting happens there.
* All arrays of a stage block share one shape. `index` is the dimension a stage acts on.
* The flow-utility function of a `LogitChoice` returns **one** array of shape
  `(n_choices, state space...)` (or anything that broadcasts to it), with `-Inf` for a
  choice that is not available in a state.
* Every outcome `x` of a stage is aggregated with the distribution at the beginning of
  that stage and becomes the block output `X`.

**Steady-state internals.** `ss.internals[block name][stage name]` holds the end-of-stage
backward variables, the stage's outcomes, `"law_of_motion"` and `"D"`, the distribution
at the beginning of that stage. The hetinput outputs are stored at
`ss.internals[block name]` directly.

**Options.** `steady_state_options`: `backward_tol` (1e-9), `backward_maxit`,
`forward_tol` (1e-10), `forward_maxit`, `warm_start` (start the next steady-state
solve from the last solution, useful inside a calibration loop).
`jacobian_options` and `impulse_linear_options`: `h` (1e-4), `twosided` (false),
`h_hetoutputs` (1e-5).

**Memory.** The Jacobian keeps one `T × (state space)` array per shocked input and
computes the expectation vectors one output at a time. The nonlinear transition keeps
only the value functions of each date and recomputes policies on the way forward.

See `src/examples/LaborChoice.jl` for a complete model and
`examples/05_labor_choice_stage_block.jl` for its use.

---

## 6. Solved blocks

A solved block hides a set of unknowns and targets inside a block, so that the rest of
the model sees a block whose outputs include the unknowns:

```julia
inner = create_model([household_rbc, firm_rbc, mkt_clearing_rbc]; name="inner")
labour_solved = solved(inner, Dict("L" => (0.5, 1.5)), ["euler"]; name="labour", solver="brentq")
```

In the steady state the block solves its targets for its unknowns. In Jacobians it
solves the linearized system internally (its own `H_U`), and in nonlinear transitions
it solves the nonlinear system at each step. The two-asset HANK model uses solved blocks
for the price Phillips curve (`pricing_solved`), the equity price (`arbitrage_solved`)
and investment with capital (`production_solved`). A solved block can also be built
with `SolvedBlock(block, name, unknowns, targets; solver, solver_kwargs)`.

---

## 7. Assembling a model

```julia
model = create_model([hh, simple(firm), simple(mkt_clearing)]; name="Krusell-Smith")
```

* `create_model` (alias `combine`) sorts the blocks topologically. The order in the list
  does not matter. A cycle is an error that names the variables involved; break it by
  making one of the variables an unknown.
* `model.inputs` are the variables no block produces (parameters, shocks, unknowns);
  `model.outputs` everything some block produces.
* Blocks can be nested: a model is a block and can be part of another model.
* **Reusing a block.** `remap(block, Dict("beta" => "beta_patient", ...))` renames inputs
  and outputs; `rename(block; suffix="_patient")` or `rename(block; name="hh2")` changes
  the block's name (blocks in one model need different names). `ks_remapped_dag()` shows
  two household types built from one block.
* `drawdag(model; exogenous, unknowns, targets)` returns the graph as Graphviz DOT source.
* Jacobians computed elsewhere can enter a model as a `JacobianDict`: `create_model`
  wraps it in a block.

---

## 8. Steady states

```julia
# evaluate a block or model at given inputs
ss = steady_state(model, calibration)

# solve for unknowns so that targets hold
ss = solve_steady_state(model, calibration, unknowns, targets; solver="brentq")
```

* `calibration` is a `Dict` of input values. `steady_state` returns a `SteadyStateDict`
  with all inputs and outputs, and the individual-level objects of heterogeneous-agent
  blocks in `ss.internals`.
* `unknowns` is a `Dict(name => value)`. The value is a starting point, a pair
  `(lower, upper)` (a bracket, for univariate solvers) or a triple `(lower, start, upper)`
  (bounds with a starting point, for multivariate solvers).
* `targets` is a `Dict(name => value)`: the output `name` must equal `value`. The value
  can be another variable's name: `Dict("B" => "Bh")` means `B = Bh`.
* **Solvers.**
  * One unknown: `"brentq"` (and the Python names `"bisect"`, `"brenth"`, `"ridder"`,
    `"toms748"`, `"newton"`, `"secant"`, `"halley"`). With a bracket `(lower, upper)`
    the root is found with Brent's method (Roots.jl); with a single starting value, with
    Newton's method.
  * Several unknowns: `"broyden_custom"` (alias `"broyden"`) and `"newton_custom"`, the
    toolbox's own Broyden and Newton solvers with backtracking; or `"hybr"` (and the
    Python names `"lm"`, `"broyden1"`, `"broyden2"`, `"anderson"`, ...), all of which call
    NLsolve's default trust-region method.
  * If no solver is given, `provide_solver_default` picks `"brentq"` for one bracketed
    unknown and `"broyden_custom"` for several.
* Bounds on multivariate unknowns are enforced by a linear continuation of the residual
  outside the box (`constrained_method="linear_continuation"`).
* `solve_steady_state` warns if the largest target residual at the returned point is
  above 1e-5.
* **Two models for one steady state.** Often the dynamic model and the steady-state
  model differ: in the steady state some prices are calibrated and some parameters
  solved for. The examples build a separate steady-state model (`model_ss`) with blocks
  that invert the steady-state relationships, solve it, and use the result with the
  dynamic model. `steady_state(model, ss)` then checks that the dynamic model is at
  rest at that point.
* `dissolve = [name of a solved block]` treats that solved block's unknowns as given
  calibration inputs instead of solving them.

---

## 9. Jacobians

```julia
J = jacobian(hh, ss, ["r", "w"], ["A", "C"]; T=300)      # partial equilibrium
G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=300)   # general equilibrium
```

* `J["C"]["r"]` is the `T × T` matrix of the response of `C_t` to `r_s`.
* `G[output][shock]` is the general-equilibrium Jacobian. `G` contains every output of
  the model and the unknowns, so `G["Y"]["Z"] * dZ` is the response of `Y` to the shock
  path `dZ`.
* `partial_jacobians(model, ss, inputs, outputs; T)` returns the Jacobian of each block,
  keyed by block name. Pass them back as `Js=...` to any later call to avoid
  recomputing them (for example when only a simple block's parameters change).
* `H_U_factored` (a `FactoredJacobianDict` of `H_U`) can be passed to `solve_jacobian`,
  `solve_impulse_linear` and `solve_impulse_nonlinear` to reuse the factorization.
* For heterogeneous-agent blocks the Jacobian is computed with the fake-news algorithm:
  one backward step per input with finite differences (step `h`), expectation vectors
  from the steady-state law of motion, and the recursion `J = F + J(-1, -1)`.

---

## 10. Linear impulse responses

```julia
irf = solve_impulse_linear(model, ss, unknowns, targets, Dict("rstar" => dr))
irf["Y"]                                    # response of Y, length T
pe  = impulse_linear(hh, ss, Dict("r" => dr))   # partial equilibrium, one block
```

`solve_impulse_linear` returns an `ImpulseDict` with the shocks, the unknowns and every
scalar output. It reuses block Jacobians passed through `Js`. For many shocks, computing
`G` once with `solve_jacobian` and multiplying is cheaper.

Convenience wrapper: `impulse_linear(model, ss, inputs, unknowns, targets, exog_names; T)`
returns a `Dict` of responses and `G` together; a scalar shock is held constant for `T`
periods and a short vector is padded with zeros.

---

## 11. Nonlinear transitions

```julia
nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("rstar" => dr);
                             tol=1e-8, maxit=30, verbose=true, warm_start=false,
                             internals=Dict(hh.name => ["D", "Dbeg_terminal"]))
```

* This is the perfect-foresight response to an unanticipated shock at date 0 (an "MIT
  shock"). The economy starts at the steady state and must return to it by date `T`.
* The solver iterates `U(k+1) = U(k) - λ H_U^{-1} H(U(k), Z)` with the steady-state
  Jacobian `H_U`. The step length `λ ∈ {1, 1/2, 1/4, ...}` is accepted only if the largest
  target residual falls (`backtrack=true`, `max_backtracks=25`). `backtrack=false` takes
  full steps, as the Python toolbox does. `warm_start=true` starts from the linear
  solution, which helps for large shocks.
* `internals` asks heterogeneous-agent blocks to report individual-level paths:
  `"D"` (distribution after the exogenous shock, `T × state space`), `"Dbeg"`,
  `"Dbeg_terminal"` (beginning-of-period distribution after the last date), and any
  individual outcome (`"a"`, `"c"`). They are in `nl.internals[block name]`.
* `terminal_distribution_distance(nl, hh.name)` tells whether `T` is long enough: the
  distribution after the last date should be close to the steady state.
* `ss_initial` starts the transition from a different steady state (a transition between
  steady states).
* `impulse_nonlinear(block, ss, inputs)` evaluates one block (or a model at given paths
  of the unknowns) without solving for equilibrium.
* Convenience wrapper: `td_nonlinear(model, ss, inputs, unknowns, targets; T, tol, maxit)`.

---

## 12. Results: SteadyStateDict, ImpulseDict, JacobianDict

**`SteadyStateDict`.** `ss["Y"]` is a value; `ss.toplevel` the aggregate values;
`ss.internals[block name]` the individual-level objects of a heterogeneous-agent block
(`"D"`, `"Dbeg"`, the Markov matrices, the backward variables and the policies).
Array-valued hetinput and hetoutput results (grids, income, individual hetoutputs) are
stored at the top level. `SequenceJacobian.extract_ss_dict(hh, ss)` returns one
dictionary with both. `ss[["Y", "C"]]`
returns a sub-dictionary; `merge!` and `copy` work as for dictionaries.

**`ImpulseDict`.** `irf["Y"]` is a path of length `irf.T`. `ImpulseDict`s support
`+`, `-`, `*` and `/` with each other and with numbers, `∪` (union), sub-selection
`irf[["Y", "C"]]`, and `irf.internals` for individual-level paths. `pack(irf)` stacks
the paths into one vector (alphabetical order unless `force_order` is given);
`unpack(v, names, T)` reverses it. `get(irf, names)` returns the selected paths.

**`JacobianDict`.** `J["C"]["r"]` is a Jacobian (a matrix, or a `SimpleSparse` lead/lag
operator for simple blocks; `make_matrix(x, T)` turns either into a matrix).
`J[outputs, inputs]` selects a sub-Jacobian. Algebra:

| Operation | Meaning |
|---|---|
| `compose(J, K)` or `J * K` with `K` a `JacobianDict` | chain rule: `J ∘ K` |
| `apply(J, shocks)` or `J * shocks` with a `Dict`/`ImpulseDict` | responses to the shock paths |
| `pack(J; T)` / `unpack(M, outputs, inputs, T)` | one big matrix and back |
| `jacobian_identity(names)` | the identity on `names` |
| `addinputs(J)` | add identity entries for inputs that are not outputs |
| `remap(J, Bijection(...))` | rename variables |
| `FactoredJacobianDict(J; T)` | LU factorization of a square `JacobianDict`, for repeated solves |

---

## 13. Options and accuracy

Every block carries default options for each method, in dictionaries
`steady_state_options`, `solve_steady_state_options`, `impulse_nonlinear_options`,
`solve_impulse_nonlinear_options`, `impulse_linear_options`, `jacobian_options`. They
can be overridden, in increasing priority:

```julia
hh.steady_state_options["backward_tol"] = 1e-10                         # on the block
ss = solve_steady_state(model, cal, unknowns, targets;
                        options=Dict("hh" => Dict("backward_tol" => 1e-10)))  # per block in a call
ss = steady_state(model, cal; backward_tol=1e-10, forward_tol=1e-12)    # keyword: all blocks that know it
```

Keyword arguments travel down through combined and solved blocks to every block that
has the option. Option names may be Strings or Symbols.

**HetBlock defaults.** Steady state: `backward_tol=1e-8`, `backward_maxit=10_000`,
`forward_tol=1e-10`, `forward_maxit=100_000`. Jacobian: `h=1e-4`, `twosided=false`.
Nonlinear transition: `monotonic=false`.

**Accuracy of household Jacobians.** The default differentiates the backward step, the
hetinputs and the hetoutputs one-sided with step `1e-4`, exactly as Python does. For a
strongly nonlinear household this is accurate to three or four digits. Two-sided
differences are much more accurate at twice the cost:

```julia
hh = het_blocks(model)[1]
G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=300,
                   options=Dict(hh.name => Dict("twosided" => true)))
```

(A bare `twosided=true` keyword on `solve_jacobian` applies to the model block itself
and does not reach the household block; use `options` or set it on the block.)
`fake_news_error(hh, ss, inputs, outputs; h=1e-4, h_direct=1e-6, twosided=...)`
measures the error for your own block.

**Convergence of the distribution.** The forward iteration stops when no element of the
distribution changes by more than `forward_tol` in one step. If the distribution has a
slow part (a state that changes rarely), this says little about the distance from the
limit. Tighten `forward_tol` (to 1e-12 or 1e-14) and check that the aggregates you care
about stop moving.

---

## 14. Estimation

Around the steady state, the model's response to a unit innovation in a shock is its
moving-average representation. With `M[t, o, z]` the response of observable `o` at
date `t` to a unit innovation in shock `z`:

```julia
M = build_MA_representation(G, ["Y", "pi"], ["rstar"], T)   # i.i.d. shocks: first columns of G
Sigma = all_covariances(M, sigmas)                           # Sigma[l+1, i, j] = Cov(x_i(t), x_j(t+l))
L = log_likelihood(Y, Sigma; sigma_measurement=me)           # Y: Tobs × O data matrix
```

For an AR(1) shock with persistence `ρ`, use `G[obs][shock] * ρ.^(0:T-1)` as the
response (see `examples/07_estimation.jl`). `all_covariances` uses the FFT, as in
Python. `build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)` and
`log_likelihood_formula(y, V)` are the two steps of `log_likelihood`. Jacobians of
heterogeneous-agent blocks do not depend on parameters of other blocks, so in an
estimation loop pass them through `Js` and recompute only what changes.

---

## 15. Validation and household statistics

These tools are not in the Python toolbox. They follow checks in Yuxuan Zhao's
sequence-space notebooks, written here for arbitrary blocks.

| Function | Use |
|---|---|
| `steady_path_error(model, ss, unknowns, targets; T=20)` | Feed the steady state through the nonlinear transition. The targets should stay at the level of the steady-state tolerances; anything larger is a timing error or an unconverged steady state. |
| `fake_news_error(block, ss, inputs, outputs; T=10, h, h_direct, twosided)` | Compare the block's Jacobian with a brute-force one (`jacobian_direct`). |
| `terminal_distribution_distance(irf, block name)`, `distribution_distance(model, ss, paths)` | Is `T` long enough for the distribution to return to the steady state? |
| `channel_decomposition(model, ss, G, shocks; output="C", direct=["r"])` | Split the GE response of a household aggregate by household input: `dO = Σ_x J^{O,x} dx`. Inputs in `direct` are summed into the direct effect. |
| `mpc_statistics(hh, ss; transfer, amount, consumption="c", horizons=(1, 2, 4), groups)` | Marginal propensities to consume out of a one-time transfer: impact and cumulative, mean, median, 90th percentile, share above 0.5, group means. |
| `het_blocks(model)`, `weighted_quantile(values, weights, p)` | Helpers. |

`examples/06_validation_and_household_statistics.jl` uses all of them.

---

## 16. The test models

| Function | Returns | Model |
|---|---|---|
| `rbc_dag()` | `model, ss, unknowns, targets, exogenous` | RBC with labor (Python `examples.rbc`) |
| `ks_dag()` | `model_ss, ss, model, unknowns, targets, exogenous` | Krusell-Smith (`examples.krusell_smith`) |
| `ks_remapped_dag()` | same | Krusell-Smith with patient and impatient households |
| `hank_dag()` | same | One-asset HANK with endogenous labor (`examples.hank`) |
| `two_asset_dag()` | same | Two-asset HANK (`examples.two_asset`); steady state takes minutes |
| `labor_choice_dag()` | same | Krusell-Smith with a logit work/home choice, written as a `StageBlock` (this port) |

The model functions are in `src/examples/`. They are worth reading as templates.
Keyword arguments replace calibration entries: `ks_dag(nS=7, nA=500)` solves the
Krusell-Smith model on the grids of the Python notebook instead of the small test grids.

---

## 17. Troubleshooting

* **"... is output of both block i and block j"**: two blocks produce the same name.
  Rename one output, or `remap` one block.
* **A cycle in the DAG**: a variable depends on itself within the period. Make one of
  the variables in the cycle an unknown and add a target that pins it down.
* **A block has no outputs or the wrong ones**: the output names come from the
  function's `return` statement. Return a NamedTuple, `return (x=x,)`, and define the
  function in a file.
* **`MethodError: no method matching log(::Displace)`** (or `AccumulatedDerivative`):
  in a simple block write `apply(x, log)` instead of `log(x)`.
* **The steady-state solver stops at a bad point**: check the warning about the largest
  residual, give a bracket for univariate problems, or bounds `(lower, start, upper)`.
* **The nonlinear solver does not converge**: try `warm_start=true`, a smaller shock,
  a larger `T`, or `verbose=true` to see which target fails.
* **The first call is slow**: Julia compiles each function on first use. Later calls
  with the same types are fast.
* **"ambiguous" names after including the source twice**: restart Julia and load the
  package once with `using SequenceJacobian`.
