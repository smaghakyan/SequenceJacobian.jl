# Two-Asset HANK Model
# Julia port of Python SSJ examples/two_asset.py and hetblocks/hh_twoasset.py

# ============================================================
# Part 1: Adjustment cost functions
# ============================================================

# ============================================================
# Part 2: Two-asset household functions
# ============================================================

# ============================================================
# Part 3: Simple block functions
# ============================================================




function labor_f(; Y, w, K, Z, alpha)
    N  = (Y / Z / K(-1)^alpha)^(1/(1-alpha))
    mc = w * N / (1-alpha) / Y
    return (N=N, mc=mc)
end

function investment_f(; Q, K, r, N, mc, Z, delta, epsI, alpha)
    inv = (K / K(-1) - 1) / (delta*epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1)/K)^(1-alpha) * mc(+1) -
          (K(+1)/K - (1-delta) + (K(+1)/K - 1)^2 / (2*delta*epsI)) +
          K(+1)/K * Q(+1) - (1+r(+1)) * Q
    return (inv=inv, val=val)
end

function dividend_f(; Y, w, N, K, pi, mup, kappap, delta, epsI)
    psip     = mup/(mup-1)/2/kappap * apply(1+pi, log)^2 * Y
    k_adjust = K(-1) * (K/K(-1) - 1)^2 / (2*delta*epsI)
    I        = K - (1-delta)*K(-1) + k_adjust
    div      = Y - w*N - I - psip
    return (psip=psip, I=I, div=div)
end


function fiscal_f(; r, w, N, G, Bg)
    tax = (r*Bg + G)/w/N
    return (tax=tax,)
end

function finance_f(; i, p, pi, r, div, omega, pshare)
    rb     = r - omega
    ra     = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return (rb=rb, ra=ra, fisher=fisher)
end


function union_f(; piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta)
    wnkpc = kappaw*(vphi*N^(1+1/frisch) - (1-tax)*w*N*UCE/muw) +
            beta * apply(1+piw(+1), log) - apply(1+piw, log)
    return (wnkpc=wnkpc,)
end

function mkt_clearing_f(; p, A, B, Bg, C, I, G, CHI, psip, omega, Y)
    wealth    = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + psip + omega*B - Y
    return (asset_mkt=asset_mkt, wealth=wealth, goods_mkt=goods_mkt)
end


function partial_ss_f(; Y, N, K, r, tot_wealth, Bg, delta)
    p     = tot_wealth - Bg
    mc    = 1 - r*(p - K)/Y
    mup   = 1/mc
    alpha = (r + delta)*K/Y/mc
    Z     = Y * K^(-alpha) * N^(alpha-1)
    w     = mc*(1-alpha)*Y/N
    return (p=p, mc=mc, mup=mup, alpha=alpha, Z=Z, w=w)
end

function union_ss_f(; tax, w, UCE, N, muw, frisch)
    vphi  = (1-tax)*w*UCE/muw/N^(1+1/frisch)
    wnkpc = vphi*N^(1+1/frisch) - (1-tax)*w*UCE/muw
    return (vphi=vphi, wnkpc=wnkpc)
end

# ============================================================
# Part 4: Grid and income hetinputs
# ============================================================

function make_grids_twoasset(; bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z)
    b_grid = agrid(Float64(bmax), Int(nB))
    a_grid = agrid(Float64(amax), Int(nA))
    k_grid = reverse(agrid(Float64(kmax), Int(nK)))
    e_grid, _, Pi = markov_rouwenhorst(rho_z, sigma_z, Int(nZ))
    return (b_grid=b_grid, a_grid=a_grid, k_grid=k_grid, e_grid=e_grid, Pi=Pi)
end

function income_twoasset(; e_grid, tax, w, N)
    z_grid = (1-tax)*w*N .* e_grid
    return (z_grid=z_grid,)
end

# ============================================================
# Part 5: DAG assembly
# ============================================================

# Build and solve the Two-Asset HANK model.
# Returns (ss_model, ss, td_model, unknowns, targets, exogenous).

function pricing_f(; pi, mc, r, Y, kappap, mup)
    nkpc = kappap * (mc - 1/mup) + Y(+1)/Y * apply(1+pi(+1), log) / (1+r(+1)) - apply(1+pi, log)
    return (nkpc=nkpc,)
end

function arbitrage_f(; div, p, r)
    equity = div(+1) + p(+1) - p * (1+r(+1))
    return (equity=equity,)
end

function taylor_f(; rstar, pi, phi)
    i = rstar + phi * pi
    return (i=i,)
end

function wage_f(; pi, w)
    piw = (1+pi) * w / w(-1) - 1
    return (piw=piw,)
end

function share_value_f(; p, tot_wealth, Bh)
    pshare = p / (tot_wealth - Bh)
    return (pshare=pshare,)
end

function two_asset_dag(; calibration_overrides...)
    pricing_b    = simple(pricing_f)
    arbitrage_b  = simple(arbitrage_f)
    labor_b      = simple(labor_f)
    investment_b = simple(investment_f)
    dividend_b   = simple(dividend_f)
    taylor_b     = simple(taylor_f)
    fiscal_b     = simple(fiscal_f)
    finance_b    = simple(finance_f)
    wage_b       = simple(wage_f)
    union_b      = simple(union_f)
    mkt_b        = simple(mkt_clearing_f)
    share_b      = simple(share_value_f)
    partial_ss_b = simple(partial_ss_f)
    union_ss_b   = simple(union_ss_f)

    hh_block = add_hetinputs(hh_twoasset, [income_twoasset, make_grids_twoasset])

    calibration = Dict{String,Any}(
        "Y"=>1., "N"=>1., "K"=>10., "r"=>0.0125, "rstar"=>0.0125, "tot_wealth"=>14.,
        "delta"=>0.02, "pi"=>0., "kappap"=>0.1, "muw"=>1.1, "Bh"=>1.04,
        "Bg"=>2.8, "G"=>0.2, "eis"=>0.5, "frisch"=>1., "chi0"=>0.25, "chi2"=>2.,
        "epsI"=>4., "omega"=>0.005, "kappaw"=>0.1, "phi"=>1.5,
        "Q"=>1.,
        # grids of the Python example (nB=50, nA=70, nK=50)
        "nZ"=>3, "nB"=>50, "nA"=>70, "nK"=>50,
        "bmax"=>50., "amax"=>4000., "kmax"=>1., "rho_z"=>0.966, "sigma_z"=>0.92
    )

    two_asset_ss = create_model(
        [hh_block, partial_ss_b, union_ss_b, dividend_b, taylor_b, fiscal_b,
         share_b, finance_b, mkt_b]; name="Two-Asset HANK SS")

    _override_calibration!(calibration, calibration_overrides)

    unknowns_ss = Dict("beta"=>0.976, "chi1"=>6.5)
    targets_ss  = Dict("asset_mkt"=>0., "B"=>"Bh")

    cali = solve_steady_state(two_asset_ss, calibration, unknowns_ss, targets_ss;
                              solver="broyden", verbose=true, ttol=1e-8)

    # Build SolvedBlocks
    pricing_solved   = SolvedBlock(pricing_b, "pricing_solved",
                                   Dict("pi" => (-0.1, 0.1)), ["nkpc"]; solver="brentq")
    arbitrage_solved = SolvedBlock(arbitrage_b, "arbitrage_solved",
                                   Dict("p" => (5., 15.)), ["equity"]; solver="brentq")
    production       = create_model([labor_b, investment_b]; name="production")
    production_solved = SolvedBlock(production, "production_solved",
                                    Dict("Q"=>1., "K"=>10.), ["inv","val"]; solver="broyden")

    # TD model — use SolvedBlocks, NOT the plain blocks
    two_asset_td = create_model(
        [hh_block, pricing_solved, arbitrage_solved, production_solved, dividend_b,
         taylor_b, fiscal_b, share_b, finance_b, wage_b, union_b, mkt_b];
        name="Two-Asset HANK")

    ss = steady_state(two_asset_td, cali)

    unknowns  = ["r", "w", "Y"]
    targets   = ["asset_mkt", "fisher", "wnkpc"]
    exogenous = ["rstar", "Z", "G"]

    return two_asset_ss, ss, two_asset_td, unknowns, targets, exogenous
end