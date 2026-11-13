# LaborChoice.jl: a test model for StageBlock.
#
# The Python toolbox has no packaged example model built from stages, so this one is
# added here to exercise StageBlock in steady state, Jacobians and transitions. It is a
# Krusell-Smith/Aiyagari economy in which households also decide each period whether to
# work. A period has three stages:
#
#   1. "prod"   ExogenousMaker: idiosyncratic productivity z moves (dimension 2)
#   2. "labor"  LogitChoice:    choose s in {home, work} (dimension 1), with
#                               type-1 extreme-value taste shocks of scale `taste_scale`
#   3. "consav" Continuous1D:   consume and save, assets a (dimension 3)
#
# The state space is (s, z, a). Workers earn w*z and pay a utility cost vphi; households
# at home produce b units of the good. Firms are competitive with a Cobb-Douglas
# technology in capital and efficiency units of labor.

function _lc_grids(; rho_z, sigma_z, nZ, amax, nA)
    z_grid, _, Pi_z = markov_rouwenhorst(rho_z, sigma_z, Int(nZ))
    a_grid = agrid(amax, Int(nA))
    return (z_grid=z_grid, Pi_z=Pi_z, a_grid=a_grid)
end

# income and indicators on the (s, z) part of the state space; s = 1 home, s = 2 work
function _lc_income(; w, b, z_grid)
    work = reshape([0.0, 1.0], 2, 1, 1)
    ne_grid  = work .* reshape(z_grid, 1, :, 1)                 # efficiency units supplied
    hp_grid  = (1 .- work) .* b .+ 0 .* ne_grid                 # home production
    emp_grid = work .+ 0 .* ne_grid                             # employment indicator
    y = w .* ne_grid .+ hp_grid
    return (y=y, ne_grid=ne_grid, hp_grid=hp_grid, emp_grid=emp_grid)
end

_lc_util(c, eis) = eis == 1 ? log(c) : c^(1 - 1/eis) / (1 - 1/eis)

function _lc_init(; a_grid, y, r, eis, beta)
    coh = (1 + r) .* reshape(a_grid, 1, 1, :) .+ y
    c = 0.1 .* coh
    Va = (1 + r) .* c .^ (-1/eis)
    V = _lc_util.(c, eis) ./ (1 - beta)
    return (V=V, Va=Va)
end

# flow utility of each choice d (dimension 1) at each state (s, z, a): working costs vphi
function _lc_flow(; vphi, z_grid, a_grid)
    flow = zeros(2, 2, length(z_grid), length(a_grid))
    flow[2, :, :, :] .= -vphi
    return (flow=flow,)
end

# consumption-saving by the endogenous grid method; V and Va are the end-of-stage
# (next period's beginning-of-period) value function and its derivative
function _lc_consav(; V, Va, a_grid, y, r, beta, eis, ne_grid, hp_grid, emp_grid)
    ag = reshape(a_grid, 1, 1, :)
    c_endo = max.(beta .* Va, 1e-300) .^ (-eis)
    coh = (1 + r) .* ag .+ y
    a = interpolate_y(c_endo .+ ag, coh, ag)
    a = max.(a, a_grid[1])
    c = coh .- a
    V_new  = _lc_util.(c, eis) .+ beta .* interpolate_y(ag, a, V)
    Va_new = (1 + r) .* c .^ (-1/eis)
    ne  = ne_grid .+ 0 .* a
    hp  = hp_grid .+ 0 .* a
    emp = emp_grid .+ 0 .* a
    return (V=V_new, Va=Va_new, a=a, c=c, ne=ne, hp=hp, emp=emp)
end

function _lc_firm(; K, L, Z, alpha, delta)
    r = alpha * Z * (K(-1) / L)^(alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L)^alpha
    Y = Z * K(-1)^alpha * L^(1 - alpha)
    return (r=r, w=w, Y=Y)
end

# steady state: r and w are calibrated, so capital and TFP follow from labor supply
function _lc_firm_ss(; r, w, NE, alpha, delta)
    KL = alpha / (1 - alpha) * w / (r + delta)
    K = KL * NE
    L = NE
    Z = w / ((1 - alpha) * KL^alpha)
    Y = Z * K^alpha * L^(1 - alpha)
    return (K=K, L=L, Z=Z, Y=Y)
end

function _lc_mkt_clearing(; A, K, NE, L, Y, C, HP, delta)
    asset_mkt = A - K
    labor_mkt = NE - L
    I = K - (1 - delta) * K(-1)
    goods_mkt = Y + HP - C - I
    return (asset_mkt=asset_mkt, labor_mkt=labor_mkt, I=I, goods_mkt=goods_mkt)
end

"""
    labor_choice_household()

The household `StageBlock` of the labor-choice model, with its grids and income as
hetinputs. Outputs: `A`, `C`, `NE` (efficiency units of labor), `HP` (home production)
and `EMP` (employment rate).
"""
function labor_choice_household()
    stages = [ExogenousMaker("Pi_z", 2; name="prod"),
              LogitChoice("V", ["Va"], 1, "taste_scale"; f=_lc_flow, name="labor"),
              Continuous1D(["V", "Va"], "a", _lc_consav; name="consav")]
    return StageBlock(stages; name="hh", backward_init=_lc_init, hetinputs=[_lc_grids, _lc_income])
end

"""
    labor_choice_dag(; calibration_overrides...)

Labor-choice model built from a `StageBlock`. Returns
`(model_ss, ss, model, unknowns, targets, exogenous)` with unknowns `["K", "L"]`,
targets `["asset_mkt", "labor_mkt"]` and exogenous `["Z"]`. The steady state solves for
the discount factor that clears the asset market at the calibrated interest rate.
"""
function labor_choice_dag(; calibration_overrides...)
    hh = labor_choice_household()
    model    = create_model([hh, simple(_lc_firm),    simple(_lc_mkt_clearing)]; name="Labor choice")
    model_ss = create_model([hh, simple(_lc_firm_ss), simple(_lc_mkt_clearing)]; name="Labor choice SS")

    calibration = Dict{String,Any}(
        "eis" => 1.0, "delta" => 0.025, "alpha" => 0.36,
        "rho_z" => 0.95, "sigma_z" => 0.4, "nZ" => 3, "amax" => 150, "nA" => 40,
        "r" => 0.01, "w" => 1.0, "b" => 0.3, "vphi" => 0.5, "taste_scale" => 0.2)

    _override_calibration!(calibration, calibration_overrides)
    unknowns_ss = Dict("beta" => (0.90, 0.989))
    targets_ss  = Dict("asset_mkt" => 0.0)
    ss = solve_steady_state(model_ss, calibration, unknowns_ss, targets_ss; solver="brentq")

    return model_ss, ss, model, ["K", "L"], ["asset_mkt", "labor_mkt"], ["Z"]
end
