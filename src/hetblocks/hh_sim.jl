# hh_sim: Standard Incomplete Market model.
# Julia port of Python SSJ hetblocks/hh_sim.py

function hh_init(; a_grid, y, r, eis)
    coh = max.((1+r) .* reshape(a_grid, 1, :) .+ reshape(y, :, 1), 1e-10)
    Va  = (1+r) .* (0.1 .* coh) .^ (-1/eis)
    return (Va=Va,)
end

function _hh(; Va_p, a_grid, y, r, beta, eis)
    uc_nextgrid = max.(beta .* Va_p, 1e-300)
    c_nextgrid  = max.(uc_nextgrid, 1e-300) .^ (-eis)
    coh = (1+r) .* reshape(a_grid, 1, :) .+ reshape(y, :, 1)
    a   = interpolate_y(c_nextgrid .+ reshape(a_grid, 1, :), coh, reshape(a_grid, 1, :))
    setmin(a, a_grid[1])
    c   = coh .- a
    Va  = (1+r) .* c .^ (-1/eis)
    return (Va=Va, a=a, c=c)
end

# hh: the HetBlock — matches Python's `hh` exported name
const hh_sim = Base.invokelatest(het, _hh; exogenous="Pi", policy="a", backward="Va", backward_init=hh_init)

# Grid and income process hetinputs (matches Python make_grids signature).
function make_grids(; rho_e, sd_e, n_e, min_a, max_a, n_a)
    e_grid, _, Pi = markov_rouwenhorst(rho_e, sd_e, Int(n_e))
    a_grid = asset_grid(min_a, max_a, Int(n_a))
    return (e_grid=e_grid, Pi=Pi, a_grid=a_grid)
end

function income(; w, e_grid)
    y = w .* e_grid
    return (y=y,)
end

# hh_extended: block with grids and income added as hetinputs
const hh_extended = Base.invokelatest(add_hetinputs, hh_sim, [income, make_grids])

# Example calibration matching Python's example_calibration().
function example_calibration()
    return Dict(
        "min_a"  => 0,    "max_a"  => 1000,
        "rho_e"  => 0.975, "sd_e"  => 0.7,
        "n_a"    => 200,   "n_e"   => 7,
        "w"      => 1,     "r"     => 0.01/4,
        "beta"   => 1 - 0.08/4, "eis" => 1
    )
end
