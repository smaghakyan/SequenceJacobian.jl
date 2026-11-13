# 05_labor_choice_stage_block.jl: a household built from stages (StageBlock).
#
#   julia --project=. examples/05_labor_choice_stage_block.jl
#
# The model is defined in src/examples/LaborChoice.jl. Each period has three stages:
# a productivity shock (ExogenousMaker), a work/home choice with taste shocks
# (LogitChoice), and consumption-saving (Continuous1D). Read that file to see how the
# stages are written.

using SequenceJacobian

model_ss, ss, model, unknowns, targets, exogenous = labor_choice_dag()
hh = only(b for b in model.blocks if b isa StageBlock)
println(hh)
println("Household inputs:  ", collect(hh.inputs))
println("Household outputs: ", collect(hh.outputs))
println("Steady state: beta = ", ss["beta"], ", K = ", ss["K"], ", employment rate = ", ss["EMP"])
println("Goods market residual (Walras' law): ", ss["goods_mkt"])

# individual-level objects of each stage: end-of-stage value functions, outcomes,
# "law_of_motion" and "D" (the distribution at the beginning of the stage)
internals = ss.internals[hh.name]
println("Stages: ", [s.name for s in hh.stages])
println("Objects stored for stage 'consav': ", sort(collect(keys(internals["consav"]))))

# fake-news Jacobian against brute force
fn = fake_news_error(hh, ss, ["r", "w"], ["A", "C", "NE", "EMP"]; T=6)
println("Fake-news vs brute-force Jacobian, largest difference: ", fn.max_error)

T  = 300
dZ = 0.01 .* ss["Z"] .* 0.9 .^ (0:T-1)
G  = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
for x in ["Y", "C", "K", "EMP", "w"]
    println(rpad(x, 4), "percent response, quarters 1-4: ", round.(100 .* (G[x]["Z"] * dZ)[1:4] ./ ss[x], digits=4))
end

nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => dZ); verbose=false)
println("\nImpact response of Y: linear ", (G["Y"]["Z"] * dZ)[1], ", nonlinear ", nl["Y"][1])
println("Largest goods-market residual along the nonlinear path: ", maximum(abs.(nl["goods_mkt"])))
