# 07_estimation.jl: likelihood of a model from its impulse responses
# (Python: sequence_jacobian.estimation).
#
#   julia --project=. examples/07_estimation.jl
#
# In the sequence space, the moving-average representation of a linearized model is its
# impulse responses. With an AR(1) shock to rstar in the one-asset HANK model, this
# script (1) stacks the impulse responses of output and inflation, (2) computes their
# covariances with the FFT, (3) simulates data from the model, and (4) evaluates the
# log likelihood over a grid for the shock standard deviation.

using SequenceJacobian, Random

model_ss, ss, model, unknowns, targets, exogenous = hank_dag()
T = 300
G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)

observables = ["Y", "pi"]
rho = 0.6
shock_path = rho .^ (0:T-1)                       # response to a unit innovation at date 0

# M[t, o, z]: response of observable o at date t to a unit innovation in shock z
M = zeros(T, length(observables), 1)
for (o, x) in enumerate(observables)
    M[:, o, 1] = G[x]["rstar"] * shock_path
end
# (build_MA_representation(G, observables, ["rstar"], T) does the same for i.i.d. shocks)

sigma_true = 0.0025
Sigma = all_covariances(M, [sigma_true])          # Sigma[l+1, i, j] = Cov(x_i(t), x_j(t+l))
println("Standard deviations implied by the model: Y ", sqrt(Sigma[1, 1, 1]), ", pi ", sqrt(Sigma[1, 2, 2]))

# simulate Tobs observations from the MA representation
Random.seed!(1)
Tobs = 120
innov = sigma_true .* randn(Tobs + T)
Y = zeros(Tobs, length(observables))
for t in 1:Tobs, o in 1:length(observables)
    Y[t, o] = sum(M[k, o, 1] * innov[T + t - k + 1] for k in 1:T)
end

sigma_measurement = [1e-4, 1e-4]
println("\nLog likelihood over the shock standard deviation:")
for s in (0.5, 0.75, 1.0, 1.25, 1.5) .* sigma_true
    L = log_likelihood(Y, all_covariances(M, [s]); sigma_measurement=sigma_measurement)
    println("  sigma = ", rpad(round(s, sigdigits=3), 8), " log L = ", round(L, digits=2))
end
