# 06_validation_and_household_statistics.jl: validation tools and household statistics
# on the one-asset HANK example.
#
#   julia --project=. examples/06_validation_and_household_statistics.jl
#
# Shows:
#   * the steady state fed through the nonlinear transition (timing check)
#   * the fake-news Jacobian against a brute-force Jacobian
#   * linear against nonlinear transitions, and whether T is long enough
#   * the decomposition of the consumption response into its channels
#   * marginal propensities to consume out of a one-time transfer

using SequenceJacobian

println("Building one-asset HANK model...")
hank_ss, ss, hank_model, unknowns, targets, exogenous = hank_dag()
hh = het_blocks(hank_model)[1]

T  = 300
dr = collect(-0.0025 .* 0.6 .^ (0:T-1))          # monetary easing
G  = solve_jacobian(hank_model, ss, unknowns, targets, exogenous; T=T)

# --- 1. numerical checks ---------------------------------------------------------
sp = steady_path_error(hank_model, ss, unknowns, targets; T=20)
println("Steady state fed through the transition: max target residual = ", sp.max_error)

fn = fake_news_error(hh, ss, ["r", "w", "Div", "Tax"], ["A", "C", "NE"]; T=8, h=1e-6)
println("Fake-news vs brute-force household Jacobian: max difference   = ", fn.max_error)

# --- 2. linear vs nonlinear ------------------------------------------------------
linear    = solve_impulse_linear(hank_model, ss, unknowns, targets, Dict("rstar" => dr))
nonlinear = solve_impulse_nonlinear(hank_model, ss, unknowns, targets, Dict("rstar" => dr);
                                    verbose=false, internals=Dict(hh.name => ["D", "Dbeg_terminal"]))
println("Impact response of Y: linear ", round(100 * linear["Y"][1], digits=5),
        "%, nonlinear ", round(100 * nonlinear["Y"][1], digits=5), "%")
println("Distance of the distribution from steady state after the last period: ",
        terminal_distribution_distance(nonlinear, hh.name), "  (if not small, increase T)")

# a shock five times larger, started from the linear solution
big = solve_impulse_nonlinear(hank_model, ss, unknowns, targets, Dict("rstar" => 5 .* dr);
                              verbose=false, warm_start=true, maxit=100)
println("5x shock, impact response of Y: 5 x linear ", round(500 * linear["Y"][1], digits=5),
        "%, nonlinear ", round(100 * big["Y"][1], digits=5), "%")

# --- 3. where does the consumption response come from? -------------------------
ch = channel_decomposition(hank_model, ss, G, Dict("rstar" => dr); output="C", direct=["r"])
println("\nConsumption response on impact (% of steady-state C), adding-up error ", ch.closure_error)
scale = 100 / ss["C"]
println("  total                 ", round(scale * ch.total[1], digits=5))
println("  direct (real rate r)  ", round(scale * ch.direct[1], digits=5))
for (x, c) in sort(collect(ch.contributions), by=first)
    x == "r" || println("  indirect via ", rpad(x, 8), " ", round(scale * c[1], digits=5))
end

# --- 4. MPCs out of a one-time transfer ------------------------------------------
ssd = SequenceJacobian.extract_ss_dict(hh, ss)
labor_income = sum(ssd["D"] .* (ss["w"] .* reshape(ssd["e_grid"], :, 1) .* ssd["n"]))
constrained  = ssd["a"] .<= ssd["a_grid"][1] + 1e-12
m = mpc_statistics(hh, ss; transfer="T", amount=0.03 * labor_income, consumption="c",
                   horizons=(1, 2, 4), groups=Dict("constrained" => constrained))
println("\nMPC out of a transfer of 3% of mean quarterly labor income (replay error ", m.replay_error, ")")
println("  share at the borrowing constraint: ", round(100 * m.group_mass["constrained"], digits=2), "%")
for h in (1, 2, 4)
    s = m.stats[h]
    println("  ", h, " quarter(s): mean ", round(s.mean, digits=4), ", median ", round(s.median, digits=4),
            ", constrained ", round(s.groups["constrained"], digits=4))
end
println("\n(The example uses a deliberately tiny grid, nS=2 and nA=10: these numbers show the tools, not a calibration.)")
