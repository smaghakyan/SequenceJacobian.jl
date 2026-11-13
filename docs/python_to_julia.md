# Translating a Python sequence-jacobian model

The Julia interface mirrors the Python toolbox. A model translates line by line once a
few differences in syntax are taken into account.

## Side by side

Python:

```python
import sequence_jacobian as sj

@sj.simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    ...
    return Va, a, c

household = hh.add_hetinputs([income, make_grids])
model = sj.create_model([household, firm, mkt_clearing], name="Krusell-Smith")
ss = model.solve_steady_state(calibration, {'beta': (0.98/1.01, 0.999/1.01)},
                              {'asset_mkt': 0}, solver='brentq')
G = model.solve_jacobian(ss, ['K'], ['asset_mkt'], ['Z'], T=300)
dK = G['K']['Z'] @ dZ
```

Julia:

```julia
using SequenceJacobian

function firm(; K, L, Z, alpha, delta)
    r = alpha * Z * (K(-1) / L)^(alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L)^alpha
    Y = Z * K(-1)^alpha * L^(1 - alpha)
    return (r=r, w=w, Y=Y)
end

function household(; Va_p, a_grid, y, r, beta, eis)
    ...
    return (Va=Va, a=a, c=c)
end
hh = het(household; exogenous="Pi", policy="a", backward="Va", backward_init=hh_init)

household_block = add_hetinputs(hh, [income, make_grids])
model = create_model([household_block, simple(firm), simple(mkt_clearing)]; name="Krusell-Smith")
ss = solve_steady_state(model, calibration, Dict("beta" => (0.98/1.01, 0.999/1.01)),
                        Dict("asset_mkt" => 0.0); solver="brentq")
G = solve_jacobian(model, ss, ["K"], ["asset_mkt"], ["Z"]; T=300)
dK = G["K"]["Z"] * dZ
```

## Differences

| Python | Julia | Note |
|---|---|---|
| `@sj.simple def f(x, y)` | `function f(; x, y)` then `simple(f)` | Keyword arguments only. |
| `return r, w` | `return (r=r, w=w)` | Outputs are the names of the NamedTuple. |
| `@sj.het(...)` decorator | `het(f; ...)` | Same keyword arguments. |
| `model.solve_steady_state(cal, ...)` | `solve_steady_state(model, cal, ...)` | Methods are functions with the block first. The same holds for `steady_state`, `jacobian`, `solve_jacobian`, `impulse_linear`, `solve_impulse_linear`, `impulse_nonlinear`, `solve_impulse_nonlinear`, `partial_jacobians`. |
| `hh.add_hetinputs([...])` | `add_hetinputs(hh, [...])` | Also `add_hetoutputs`, `remap`, `rename`. |
| `{'beta': 0.98}` | `Dict("beta" => 0.98)` | Strings in double quotes. |
| `['K']` | `["K"]` | |
| `x ** 2` | `x^2` | |
| `G['K']['Z'] @ dZ` | `G["K"]["Z"] * dZ` | |
| `np.log(x)` in a simple block | `apply(x, log)` | Any function other than `+ - * / ^`. |
| `0.9 ** np.arange(T)` | `0.9 .^ (0:T-1)` | Dates start at 0 in the formula, indices at 1. |
| `a[0]`, `a[-1]` | `a[1]`, `a[end]` | 1-based indexing. |
| `X.shape` | `size(X)` | |
| `np.newaxis` / broadcasting | `reshape(x, 1, :)` and dot operators (`.+`, `.*`) | |
| `utils.discretize.markov_rouwenhorst` | `markov_rouwenhorst` | All utilities are exported at the top level. |
| `utils.misc.mean(y, pi)` | `disc_mean(y, pi)` | Also `disc_variance`, `disc_std`. |
| `JacobianDict.identity(ks)` | `jacobian_identity(ks)` | |
| `sj.hetblocks.hh_sim.hh` | `hh_sim` | Also `hh_labor`, `hh_twoasset`. |
| `sj.examples.krusell_smith.dag()` | `ks_dag()` | Also `rbc_dag`, `hank_dag`, `two_asset_dag`. |
| `ss.internals['hh']['D']` | `ss.internals["hh"]["D"]` | |
| stage dimension `0` | stage dimension `1` | `ExogenousMaker("Pi", 1)` acts on the first dimension. |
| `sj.drawdag(model)` | `drawdag(model)` | Returns DOT source instead of drawing. |

## Things that behave the same

* Names of options and their defaults (`backward_tol`, `forward_tol`, `h`, `twosided`,
  `tol`, `maxit`, ...).
* Timing conventions: lags take the initial steady state, leads the terminal one; the
  exogenous transition happens at the beginning of the period.
* One-sided differencing of household blocks by default (see the guide on accuracy).
* The structure of `SteadyStateDict`, `ImpulseDict` and `JacobianDict`.

## Things that are not in the Python toolbox

New in this port:

* `ExogenousMaker(...; by=j)` and `ExogenousMaker(...; joint=true)` in stage blocks.
* `twosided=true` differences the hetinputs and hetoutputs too, not only the backward
  step.
* Jacobians of stage blocks with a `Continuous2D` stage (they fail in Python 1.0.0).

Following Yuxuan Zhao's sequence-space notebooks:

* `solve_impulse_nonlinear` backtracks on the step length and can start from the
  linear solution (`warm_start=true`).
* Validation tools and household statistics: `fake_news_error`, `steady_path_error`,
  `terminal_distribution_distance`, `channel_decomposition`, `mpc_statistics`.
