# 01_rbc.jl: the RBC example model (Python: sequence_jacobian.examples.rbc).
#
#   julia --project=. examples/01_rbc.jl        (from the repository root)
#
# Shows: steady state, general-equilibrium Jacobians, linear impulse responses,
# nonlinear perfect-foresight transitions.

using SequenceJacobian

model, ss, unknowns, targets, exogenous = rbc_dag()
println(model)
println("Unknowns: ", unknowns, "   targets: ", targets, "   exogenous: ", exogenous)
println("Steady state: Y = ", ss["Y"], ", K = ", ss["K"], ", r = ", ss["r"],
        ", goods market residual = ", ss["goods_mkt"])

T  = 300
dZ = 0.01 .* ss["Z"] .* 0.9 .^ (0:T-1)          # 1% TFP shock with persistence 0.9

# General-equilibrium Jacobians: G[output][input] is a T x T matrix
G  = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
dY = G["Y"]["Z"] * dZ
dC = G["C"]["Z"] * dZ
dK = G["K"]["Z"] * dZ
println("\nLinear responses (deviations from steady state, first 5 quarters)")
println("  Y: ", round.(dY[1:5], digits=6))
println("  C: ", round.(dC[1:5], digits=6))
println("  K: ", round.(dK[1:5], digits=6))

# The same responses without forming G
lin = solve_impulse_linear(model, ss, unknowns, targets, Dict("Z" => dZ))
println("  max |solve_impulse_linear - G * dZ| for Y: ", maximum(abs.(lin["Y"] .- dY)))

# Nonlinear transition for a shock ten times larger
nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => 10 .* dZ); verbose=true)
println("\nImpact response of Y to a 10% TFP shock: linear ", 10 * dY[1], ", nonlinear ", nl["Y"][1])

# The DAG of the model as Graphviz DOT source (paste it into any Graphviz viewer)
println("\n", drawdag(model; exogenous=exogenous, unknowns=unknowns, targets=targets))
