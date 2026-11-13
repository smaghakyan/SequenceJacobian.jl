# 03_one_asset_hank.jl: the one-asset HANK example model (Python: examples.hank).
#
#   julia --project=. examples/03_one_asset_hank.jl
#
# Shows: a New Keynesian model with a household block that also chooses labor
# (hh_labor), a monetary policy shock, linear against nonlinear responses.

using SequenceJacobian

model_ss, ss, model, unknowns, targets, exogenous = hank_dag()
println(model)
println("Unknowns: ", unknowns, "   targets: ", targets, "   exogenous: ", exogenous)
println("Steady state: beta = ", ss["beta"], ", vphi = ", ss["vphi"], ", Y = ", ss["Y"])

T  = 300
dr = -0.0025 .* 0.6 .^ (0:T-1)                 # 25 basis point cut in rstar, persistence 0.6
G  = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
for x in ["Y", "C", "pi", "w", "r"]
    println(rpad(x, 3), " first 4 quarters: ", round.((G[x]["rstar"] * dr)[1:4], digits=6))
end

nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("rstar" => dr); verbose=false)
println("\nImpact response of Y: linear ", (G["Y"]["rstar"] * dr)[1], ", nonlinear ", nl["Y"][1])

# helper for percentage responses
println("Output response in percent, quarters 1-4: ", round.(irf_pct(G, "Y", "rstar", dr; ss_val=ss["Y"])[1:4], digits=4))
