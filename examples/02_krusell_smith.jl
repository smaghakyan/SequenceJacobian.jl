# 02_krusell_smith.jl: the Krusell-Smith example model (Python: examples.krusell_smith)
# and its variant with two household types that differ in patience.
#
#   julia --project=. examples/02_krusell_smith.jl
#
# Shows: a heterogeneous-agent block, its partial-equilibrium Jacobian (fake-news
# algorithm), general equilibrium, the household's nonlinear response with the
# distribution path kept, and remapping/renaming blocks.

using SequenceJacobian

model_ss, ss, model, unknowns, targets, exogenous = ks_dag()
hh = het_blocks(model)[1]
println(hh)
println("Household inputs: ", collect(hh.inputs))
println("Household outputs: ", collect(hh.outputs))
println("Steady state: beta = ", ss["beta"], ", K = ", ss["K"], ", asset market residual = ", ss["asset_mkt"])

T = 300
# partial equilibrium: household Jacobians with respect to prices
J = jacobian(hh, ss, ["r", "w"], ["A", "C"]; T=T)
println("\nFirst row of J[C][r] (consumption today on interest rates at dates 0..4): ",
        round.(J["C"]["r"][1, 1:5], digits=5))

# general equilibrium
dZ = 0.01 .* ss["Z"] .* 0.9 .^ (0:T-1)
G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
println("GE response of K to a 1% TFP shock, quarters 1-5: ", round.((G["K"]["Z"] * dZ)[1:5], digits=6))

# household block alone, nonlinear, keeping the path of the distribution
dr = 0.001 .* 0.8 .^ (0:T-1)
pe = impulse_nonlinear(hh, ss, Dict("r" => dr); internals=Dict(hh.name => ["D", "Dbeg_terminal"]))
println("PE nonlinear response of A to a path of r, quarters 1-5: ", round.(pe["A"][1:5], digits=6))
println("Size of the stored distribution path: ", size(pe.internals[hh.name]["D"]))
println("Distance of the distribution from steady state after T: ", terminal_distribution_distance(pe, hh.name))

# two household types: rename and remap one block into two
model_ss2, ss2, model2, u2, t2, e2 = ks_remapped_dag()
println("\nTwo-type model: ", [b.name for b in het_blocks(model2)], ", beta_patient = ", ss2["beta_patient"])
G2 = solve_jacobian(model2, ss2, u2, t2, e2; T=T)
println("GE response of K, quarters 1-5: ", round.((G2["K"]["Z"] * dZ)[1:5], digits=6))
