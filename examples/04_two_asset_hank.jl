# 04_two_asset_hank.jl: the two-asset HANK example model (Python: examples.two_asset).
#
#   julia --project=. examples/04_two_asset_hank.jl
#
# The household block has two endogenous states (liquid and illiquid assets) with a
# convex portfolio adjustment cost. The steady state takes a few minutes.

using SequenceJacobian

println("Solving the two-asset steady state (this takes a few minutes)...")
model_ss, ss, model, unknowns, targets, exogenous = two_asset_dag()
println("Steady state: Y = ", ss["Y"], ", r = ", ss["r"], ", B = ", ss["B"], ", asset market residual = ", ss["asset_mkt"])

T = 300
drstar = -0.0025 .* 0.6 .^ (0:T-1)
G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
println("Peak response of Y (%): ", maximum(100 .* (G["Y"]["rstar"] * drstar)))
println("Peak response of C (%): ", maximum(100 .* (G["C"]["rstar"] * drstar)))

# Jacobians of strongly nonlinear household blocks are more accurate with two-sided
# differences (about twice the cost); the default reproduces Python exactly.
hh = het_blocks(model)[1]
G2 = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T,
                    options=Dict(hh.name => Dict("twosided" => true)))
println("Peak response of Y (%), two-sided household Jacobian: ", maximum(100 .* (G2["Y"]["rstar"] * drstar)))
