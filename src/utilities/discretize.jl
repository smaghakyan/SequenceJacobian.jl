# Grids and Markov chain methods.
# Julia port of Python SSJ utilities/discretize.py

using LinearAlgebra
using Distributions: Normal, cdf
using Statistics

function asset_grid(amin, amax, n)
    ubar = log(1 + log(1 + amax - amin))
    u_grid = range(0, ubar, length=n)
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

function agrid(amax, n; amin=0)
    pivot = abs(amin) + 0.25
    a_grid = exp10.(range(log10(amin + pivot), log10(amax + pivot), length=Int(n))) .- pivot
    a_grid[1] = amin
    return a_grid
end

function agrid_old(amax, N, amin=0, frac=1/25)
    apivot = (1 + amin)^(1-frac) * (1 + amax)^frac - 1
    a = exp10.(range(log10(amin + apivot), log10(amax + apivot), length=Int(N))) .- apivot
    a[1] = amin
    return a
end

function nonlinspace(amax, n, phi, amin=0)
    a_grid = zeros(n)
    a_grid[1] = amin
    for i in 2:n
        # (n - i + 1): with 1-based indexing the last step must divide by 1, so that
        # a_grid[n] == amax (the 0-based Python formula was copied without the shift,
        # which made the last grid point infinite)
        a_grid[i] = a_grid[i-1] + (amax - a_grid[i-1]) / (n - i + 1)^phi
    end
    return a_grid
end

# Find invariant distribution of a Markov chain by iteration.
# Matches Python: pi_new = pi @ Pi  (row-vector left-multiply).
function stationary(Pi::AbstractMatrix, pi_seed=nothing; tol=1e-11, maxit=10_000)
    _pi = isnothing(pi_seed) ? fill(1.0/size(Pi,1), size(Pi,1)) : copy(vec(pi_seed))
    for it in 1:maxit
        pi_new = vec(_pi' * Pi)
        if maximum(abs, pi_new - _pi) < tol
            return pi_new
        end
        _pi = pi_new
    end
    error("No convergence after $maxit forward iterations!")
end

# Mean of discretized random variable.
disc_mean(x, pi) = sum(pi .* x)

# Variance of discretized random variable.
disc_variance(x, pi) = sum(pi .* (x .- sum(pi .* x)).^2)

# Standard deviation of discretized random variable.
disc_std(x, pi) = sqrt(disc_variance(x, pi))

# Covariance of two discretized random variables.
disc_cov(x, y, pi) = sum(pi .* (x .- disc_mean(x, pi)) .* (y .- disc_mean(y, pi)))

# Correlation of two discretized random variables.
disc_corr(x, y, pi) = disc_cov(x, y, pi) / (disc_std(x, pi) * disc_std(y, pi))

function markov_tauchen(rho, sigma, N=7, m=3; normalize=true)
    s = range(-m, m, length=N)
    ds = s[2] - s[1]
    sd_innov = sqrt(1 - rho^2)
    d = Normal(0, sd_innov)

    Pi = Array{Float64}(undef, N, N)
    Pi[:,1]   = cdf.(d, s[1] .- rho .* s .+ ds/2)
    Pi[:,end] = 1 .- cdf.(d, s[end] .- rho .* s .- ds/2)
    for j in 2:N-1
        Pi[:,j] = cdf.(d, s[j] .- rho .* s .+ ds/2) .- cdf.(d, s[j] .- rho .* s .- ds/2)
    end

    _pi = stationary(Pi)
    s_arr = collect(s) .* (sigma / sqrt(disc_variance(collect(s), _pi)))
    y = normalize ? exp.(s_arr) ./ sum(_pi .* exp.(s_arr)) : s_arr
    return y, _pi, Pi
end

function markov_rouwenhorst(rho, sigma, N=7)
    N = Int(N)
    p = (1 + rho) / 2
    Pi = [p 1-p; 1-p p]

    for n in 3:N
        P1, P2, P3, P4 = (zeros(n,n) for _ in 1:4)
        P1[1:end-1, 1:end-1] = p .* Pi
        P2[1:end-1, 2:end]   = (1-p) .* Pi
        P3[2:end,   1:end-1] = (1-p) .* Pi
        P4[2:end,   2:end]   = p .* Pi
        Pi = P1 + P2 + P3 + P4
        Pi[2:end-1, :] ./= 2
    end

    _pi = stationary(Pi)
    s = collect(range(-1.0, 1.0, length=N)) .* (sigma / sqrt(disc_variance(collect(range(-1.0,1.0,length=N)), _pi)))
    y = exp.(s) ./ sum(_pi .* exp.(s))
    return y, _pi, Pi
end
