# 08_build_your_own_model.jl: writing your own blocks.
#
#   julia --project=. examples/08_build_your_own_model.jl
#
# Builds a Krusell-Smith economy from scratch with the same building blocks the
# packaged examples use:
#   * simple blocks from plain Julia functions (`simple`), with K(-1) for a lag and
#     K(+1) for a lead;
#   * a heterogeneous-agent block from a backward-iteration function (`het`), with grids
#     and income attached as hetinputs and an extra individual outcome as a hetoutput;
#   * a model assembled with `create_model`, its steady state, Jacobians and transitions.
#
# Conventions (as in the Python toolbox):
#   * block functions take keyword arguments only, named after the variables they use;
#   * outputs are returned as a NamedTuple, `return (r=r, w=w)`, and the names are read
#     from that return statement (define functions in a file, not at the REPL);
#   * a backward function takes next period's backward variable with the suffix `_p`
#     (`Va_p`) and returns the current one (`Va`) together with the policies;
#   * individual outcomes are lower case (`a`, `c`); their aggregates are upper case.

using SequenceJacobian

# ---- household -------------------------------------------------------------------
function household_init(; a_grid, y, r, eis)
    coh = (1 + r) .* reshape(a_grid, 1, :) .+ reshape(y, :, 1)
    Va = (1 + r) .* (0.1 .* coh) .^ (-1 / eis)
    return (Va=Va,)
end

function household(; Va_p, a_grid, y, r, beta, eis)
    uc_nextgrid = beta .* Va_p
    c_nextgrid = uc_nextgrid .^ (-eis)
    coh = (1 + r) .* reshape(a_grid, 1, :) .+ reshape(y, :, 1)
    a = interpolate_y(c_nextgrid .+ reshape(a_grid, 1, :), coh, reshape(a_grid, 1, :))
    setmin(a, a_grid[1])                       # borrowing constraint
    c = coh .- a
    Va = (1 + r) .* c .^ (-1 / eis)
    return (Va=Va, a=a, c=c)
end

function grids(; rho_e, sd_e, nE, amax, nA)
    e_grid, _, Pi = markov_rouwenhorst(rho_e, sd_e, Int(nE))
    a_grid = agrid(amax, Int(nA))
    return (e_grid=e_grid, Pi=Pi, a_grid=a_grid)
end

labor_income(; w, e_grid) = (y = w .* e_grid,)

# a hetoutput: an individual outcome computed after the backward step
function constrained(; a, a_grid)
    htm = Float64.(a .<= a_grid[1] + 1e-10)
    return (htm=htm,)
end

hh = het(household; exogenous="Pi", policy="a", backward="Va", backward_init=household_init)
hh = add_hetinputs(hh, [labor_income, grids])
hh = add_hetoutputs(hh, [constrained])

# ---- firms and markets -----------------------------------------------------------
function firm(; K, L, Z, alpha, delta)
    r = alpha * Z * (K(-1) / L)^(alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L)^alpha
    Y = Z * K(-1)^alpha * L^(1 - alpha)
    return (r=r, w=w, Y=Y)
end

function firm_steady_state(; r, Y, L, delta, alpha)
    K = alpha * Y / (r + delta)
    Z = Y / K^alpha / L^(1 - alpha)
    w = (1 - alpha) * Z * (K / L)^alpha
    return (K=K, Z=Z, w=w)
end

function mkt_clearing(; K, A, Y, C, delta)
    asset_mkt = A - K
    I = K - (1 - delta) * K(-1)
    goods_mkt = Y - C - I
    return (asset_mkt=asset_mkt, goods_mkt=goods_mkt, I=I)
end

model    = create_model([hh, simple(firm), simple(mkt_clearing)]; name="My Krusell-Smith")
model_ss = create_model([hh, simple(firm_steady_state), simple(mkt_clearing)]; name="My Krusell-Smith SS")
println(model)
println("Model inputs:  ", collect(model.inputs))
println("Model outputs: ", collect(model.outputs))

# ---- steady state: choose beta to clear the asset market ------------------------
calibration = Dict{String,Any}("eis" => 1.0, "delta" => 0.025, "alpha" => 0.11,
    "rho_e" => 0.966, "sd_e" => 0.5, "nE" => 7, "amax" => 200, "nA" => 200,
    "Y" => 1.0, "L" => 1.0, "r" => 0.01)
ss = solve_steady_state(model_ss, calibration, Dict("beta" => (0.98 / 1.01, 0.999 / 1.01)),
                        Dict("asset_mkt" => 0.0); solver="brentq")
println("\nbeta = ", ss["beta"], ", K = ", ss["K"], ", share at the constraint = ", ss["HTM"])

# individual-level steady-state objects
hh_ss = ss.internals[hh.name]
println("Stored for the household: ", sort(collect(keys(hh_ss))))
println("Size of the distribution: ", size(hh_ss["D"]))

# ---- dynamics --------------------------------------------------------------------
T = 300
dZ = 0.01 .* ss["Z"] .* 0.9 .^ (0:T-1)
G = solve_jacobian(model, ss, ["K"], ["asset_mkt"], ["Z"]; T=T)
println("\nResponse of K, quarters 1-5: ", round.((G["K"]["Z"] * dZ)[1:5], digits=6))
println("Response of the constrained share, quarters 1-5: ", round.((G["HTM"]["Z"] * dZ)[1:5], digits=6))
nl = solve_impulse_nonlinear(model, ss, ["K"], ["asset_mkt"], Dict("Z" => dZ); verbose=false)
println("Nonlinear response of K, quarters 1-5: ", round.(nl["K"][1:5], digits=6))
