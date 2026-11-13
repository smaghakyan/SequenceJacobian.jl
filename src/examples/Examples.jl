# Examples.jl: RBC, One-Asset HANK, and Krusell-Smith example models.
# Julia port of Python SSJ examples/rbc.py, hank.py, krusell_smith.py

# ============================================================
# RBC Model (rbc.py)
# ============================================================

function _firm_rbc(; K, L, Z, alpha, delta)
    r = alpha * Z * (K(-1) / L)^(alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L)^alpha
    Y = Z * K(-1)^alpha * L^(1-alpha)
    return (r=r, w=w, Y=Y)
end

function _household_rbc(; K, L, w, eis, frisch, vphi, delta)
    C = (w / vphi / L^(1/frisch))^eis
    I = K - (1-delta) * K(-1)
    return (C=C, I=I)
end

function _mkt_clearing_rbc(; r, C, Y, I, K, L, w, eis, beta)
    goods_mkt = Y - C - I
    euler     = C^(-1/eis) - beta * (1+r(+1)) * C(+1)^(-1/eis)
    walras    = C + K - (1+r) * K(-1) - w * L
    return (goods_mkt=goods_mkt, euler=euler, walras=walras)
end

# Calibration overrides for the example models: hank_dag(nS=7, nA=500) replaces those
# entries of the calibration; anything not given keeps the default.
function _override_calibration!(calibration::Dict{String,Any}, overrides)
    for (k, v) in pairs(overrides)
        haskey(calibration, string(k)) || error("'$(k)' is not a calibration entry of this model (entries: $(sort(collect(keys(calibration)))))")
        calibration[string(k)] = v
    end
    return calibration
end

function rbc_dag(; calibration_overrides...)
    firm_b        = simple(_firm_rbc)
    household_b   = simple(_household_rbc)
    mkt_clearing  = simple(_mkt_clearing_rbc)

    rbc_model = create_model([household_b, firm_b, mkt_clearing]; name="RBC")

    calibration = Dict{String,Any}(
        "eis"=>1., "frisch"=>1., "delta"=>0.025, "alpha"=>0.11, "L"=>1.)
    _override_calibration!(calibration, calibration_overrides)
    unknowns_ss = Dict("vphi"=>0.92, "beta"=>1/(1+0.01), "K"=>2., "Z"=>1.)
    targets_ss  = Dict("goods_mkt"=>0., "r"=>0.01, "euler"=>0., "Y"=>1.)

    ss = solve_steady_state(rbc_model, calibration, unknowns_ss, targets_ss; solver="hybr")

    unknowns  = ["K", "L"]
    targets   = ["goods_mkt", "euler"]
    exogenous = ["Z"]

    return rbc_model, ss, unknowns, targets, exogenous
end

# ============================================================
# One-Asset HANK (hank.py)
# ============================================================

function _firm_hank(; Y, w, Z, pi, mu, kappa)
    L   = Y / Z
    Div = Y - w*L - mu/(mu-1)/(2*kappa) * apply(1+pi, log)^2 * Y
    return (L=L, Div=Div)
end

function _monetary(; pi, rstar, phi)
    r = (1 + rstar(-1) + phi * pi(-1)) / (1+pi) - 1
    return (r=r,)
end

function _nkpc(; pi, w, Z, Y, r, mu, kappa)
    nkpc_res = kappa*(w/Z - 1/mu) + Y(+1)/Y * apply(1+pi(+1), log) / (1+r(+1)) - apply(1+pi, log)
    return (nkpc_res=nkpc_res,)
end

function _fiscal(; r, B)
    Tax = r * B
    return (Tax=Tax,)
end

function _mkt_clearing_hank(; A, NE, C, L, Y, B, pi, mu, kappa)
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * apply(1+pi, log)^2 * Y
    return (asset_mkt=asset_mkt, labor_mkt=labor_mkt, goods_mkt=goods_mkt)
end

function _nkpc_ss(; Z, mu)
    w = Z / mu
    return (w=w,)
end

function _make_grids_hank(; rho_s, sigma_s, nS, amax, nA)
    e_grid, pi_e, Pi = markov_rouwenhorst(rho_s, sigma_s, Int(nS))
    a_grid = agrid(amax, Int(nA))
    return (e_grid=e_grid, pi_e=pi_e, Pi=Pi, a_grid=a_grid)
end

function _transfers(; pi_e, Div, Tax, e_grid)
    div_rule = e_grid; tax_rule = e_grid
    div = Div / sum(pi_e .* div_rule) .* div_rule
    tax = Tax / sum(pi_e .* tax_rule) .* tax_rule
    T   = div .- tax
    return (T=T,)
end

function _wages_hank(; w, e_grid)
    we = w .* e_grid
    return (we=we,)
end

function _labor_supply(; n, e_grid)
    ne = reshape(e_grid, :, 1) .* n
    return (ne=ne,)
end

function hank_dag(; calibration_overrides...)
    firm_h    = simple(_firm_hank)
    monetary  = simple(_monetary)
    nkpc_b    = simple(_nkpc)
    fiscal_b  = simple(_fiscal)
    mkt_h     = simple(_mkt_clearing_hank)
    nkpc_ss_b = simple(_nkpc_ss)

    household = add_hetinputs(hh_labor, [_transfers, _wages_hank, _make_grids_hank])
    household = add_hetoutputs(household, [_labor_supply])

    hank_model    = create_model([household, firm_h, monetary, fiscal_b, mkt_h, nkpc_b];    name="One-Asset HANK")
    hank_model_ss = create_model([household, firm_h, monetary, fiscal_b, mkt_h, nkpc_ss_b]; name="One-Asset HANK SS")

    calibration = Dict{String,Any}(
        "r"=>0.005, "rstar"=>0.005, "eis"=>0.5, "frisch"=>0.5, "B"=>5.6,
        "mu"=>1.2, "rho_s"=>0.966, "sigma_s"=>0.5, "kappa"=>0.1, "phi"=>1.5,
        "Y"=>1., "Z"=>1., "pi"=>0., "nS"=>2, "amax"=>150, "nA"=>10)

    _override_calibration!(calibration, calibration_overrides)

    unknowns_ss = Dict("beta"=>0.986, "vphi"=>0.8)
    targets_ss  = Dict("asset_mkt"=>0., "NE"=>1.)

    ss_cali = solve_steady_state(hank_model_ss, calibration, unknowns_ss, targets_ss;
                                  solver="broyden_custom")
    ss = steady_state(hank_model, ss_cali)

    unknowns  = ["w", "Y", "pi"]
    targets   = ["asset_mkt", "goods_mkt", "nkpc_res"]
    exogenous = ["rstar", "Z"]

    return hank_model_ss, ss, hank_model, unknowns, targets, exogenous
end

# ============================================================
# Krusell-Smith (krusell_smith.py)
# ============================================================

function _firm_ks(; K, L, Z, alpha, delta)
    r = alpha * Z * (K(-1)/L)^(alpha-1) - delta
    w = (1-alpha) * Z * (K(-1)/L)^alpha
    Y = Z * K(-1)^alpha * L^(1-alpha)
    return (r=r, w=w, Y=Y)
end

function _mkt_clearing_ks(; K, A, Y, C, delta)
    asset_mkt = A - K
    I         = K - (1-delta) * K(-1)
    goods_mkt = Y - C - I
    return (asset_mkt=asset_mkt, goods_mkt=goods_mkt, I=I)
end

function _firm_ss_ks(; r, Y, L, delta, alpha)
    rk = r + delta
    K  = alpha * Y / rk
    Z  = Y / K^alpha / L^(1-alpha)
    w  = (1-alpha) * Z * (K/L)^alpha
    return (K=K, Z=Z, w=w)
end

function _make_grids_ks(; rho, sigma, nS, amax, nA)
    e_grid, _, Pi = markov_rouwenhorst(rho, sigma, Int(nS))
    a_grid = agrid(amax, Int(nA))
    return (e_grid=e_grid, Pi=Pi, a_grid=a_grid)
end

function _income_ks(; w, e_grid)
    y = w .* e_grid
    return (y=y,)
end

function ks_dag(; calibration_overrides...)
    firm_block    = simple(_firm_ks)
    firm_ss_block = simple(_firm_ss_ks)
    mkt_block     = simple(_mkt_clearing_ks)

    household = add_hetinputs(hh_sim, [_income_ks, _make_grids_ks])

    ks_model    = create_model([household, firm_block,    mkt_block]; name="Krusell-Smith")
    ks_model_ss = create_model([household, firm_ss_block, mkt_block]; name="Krusell-Smith SS")

    calibration = Dict{String,Any}(
        "eis"=>1., "delta"=>0.025, "alpha"=>0.11, "rho"=>0.966, "sigma"=>0.5,
        "Y"=>1., "L"=>1., "nS"=>2, "nA"=>10, "amax"=>200, "r"=>0.01)

    _override_calibration!(calibration, calibration_overrides)

    unknowns_ss = Dict("beta"=>(0.98/1.01, 0.999/1.01))
    targets_ss  = Dict("asset_mkt"=>0.)

    ss = solve_steady_state(ks_model_ss, calibration, unknowns_ss, targets_ss; solver="brentq")

    unknowns  = ["K"]
    targets   = ["asset_mkt"]
    inputs    = ["Z"]

    return ks_model_ss, ss, ks_model, unknowns, targets, inputs
end

function _aggregate_ks(; A_patient, A_impatient, C_patient, C_impatient, mass_patient)
    C = mass_patient * C_patient + (1-mass_patient) * C_impatient
    A = mass_patient * A_patient + (1-mass_patient) * A_impatient
    return (C=C, A=A)
end

function ks_remapped_dag(; calibration_overrides...)
    firm_block    = simple(_firm_ks)
    firm_ss_block = simple(_firm_ss_ks)
    mkt_block     = simple(_mkt_clearing_ks)
    agg_block     = simple(_aggregate_ks)

    base_hh  = add_hetinputs(hh_sim, [_income_ks, _make_grids_ks])
    to_map   = vcat(["beta"], collect(base_hh.outputs))
    hh_patient   = remap(rename(base_hh; suffix="_patient"),   Dict(k => k*"_patient"   for k in to_map))
    hh_impatient = remap(rename(base_hh; suffix="_impatient"), Dict(k => k*"_impatient" for k in to_map))

    ks_rem    = create_model([hh_patient, hh_impatient, firm_block,    mkt_block, agg_block]; name="KS-beta-het")
    ks_rem_ss = create_model([hh_patient, hh_impatient, firm_ss_block, mkt_block, agg_block]; name="KS-beta-het")

    calibration = Dict{String,Any}(
        "eis"=>1., "delta"=>0.025, "alpha"=>0.3, "rho"=>0.966, "sigma"=>0.5,
        "Y"=>1., "L"=>1., "nS"=>3, "nA"=>100, "amax"=>1000,
        "beta_impatient"=>0.985, "mass_patient"=>0.5,
        "r"=>0.01)   # the steady-state firm block needs r (it was missing, so this example could not run)

    _override_calibration!(calibration, calibration_overrides)

    unknowns_ss = Dict("beta_patient"=>(0.98/1.01, 0.999/1.01))
    targets_ss  = Dict("asset_mkt"=>0.)
    ss = solve_steady_state(ks_rem_ss, calibration, unknowns_ss, targets_ss; solver="brentq")

    unknowns  = ["K"]
    targets   = ["asset_mkt"]
    exogenous = ["Z"]

    return ks_rem_ss, ss, ks_rem, unknowns, targets, exogenous
end
