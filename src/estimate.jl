# Functions for calculating the log likelihood of a model from its impulse responses.
# Julia port of Python SSJ estimation.py

using FFTW
using LinearAlgebra

# Compute covariance function between O variables up to T-1 lags using FFT.
#
# Parameters
# ----------
# M      : Array{Float64,3} of shape (T, O, Z) — impulse responses of O variables to Z shocks
# sigmas : Vector{Float64} of length Z — standard deviations of shocks
#
# Returns
# -------
# Sigma  : Array{Float64,3} of shape (T, O, O) — covariance function for lags 0,...,T-1
function all_covariances(M::Array{Float64,3}, sigmas::Vector{Float64})
    T = size(M, 1)
    # Pad to length 2T-2 for circular convolution
    M_padded = zeros(2*T-2, size(M,2), size(M,3))
    M_padded[1:T, :, :] = M
    # DFT along time dimension
    dft = rfft(M_padded, 1)
    # Weight by variance
    sigma_sq = reshape(sigmas.^2, 1, 1, length(sigmas))
    total = conj.(dft) .* sigma_sq
    # Cross products at each frequency
    result_freq = zeros(ComplexF64, size(dft,1), size(M,2), size(M,2))
    for i in axes(dft, 1)
        result_freq[i, :, :] = total[i, :, :] * permutedims(dft[i, :, :], [2,1])
    end
    result = irfft(result_freq, 2*T-2, 1)
    return result[1:T, :, :]
end



# Given second moments, compute log-likelihood of data Y.
#
# Parameters
# ----------
# Y                  : Matrix{Float64} of shape (Tobs, O)
# Sigma              : Array{Float64,3} of shape (T, O, O) — from all_covariances
# sigma_measurement  : (optional) Vector{Float64} of length O, std of measurement error
#
# Returns
# -------
# L : scalar log-likelihood
function log_likelihood(Y::Matrix{Float64}, Sigma::Array{Float64,3};
                        sigma_measurement::Union{Vector{Float64},Nothing}=nothing)
    Tobs, n0 = size(Y)
    if isnothing(sigma_measurement)
        sigma_measurement = zeros(n0)
    end
    V = build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)
    y = vec(Y)
    return log_likelihood_formula(y, V)
end

# Multivariate normal log-likelihood: -log|V|/2 - y'V^{-1}y/2
function log_likelihood_formula(y::Vector{Float64}, V::Matrix{Float64})
    V_chol          = cholesky(Symmetric(V))
    quadratic_form  = dot(y, V_chol \ y)
    log_determinant = 2 * sum(log.(diag(V_chol.L)))
    return -(log_determinant + quadratic_form) / 2
end

# Assemble (Tobs*O) x (Tobs*O) full covariance matrix from lag-covariance array Sigma,
# adding measurement error variance on the diagonal.
#
# Parameters
# ----------
# Sigma              : Array{Float64,3} of shape (T, O, O)
# sigma_measurement  : Vector{Float64} of length O
# Tobs               : Int
function build_full_covariance_matrix(Sigma::Array{Float64,3},
                                      sigma_measurement::Vector{Float64},
                                      Tobs::Int)
    T = size(Sigma, 1)
    O = size(Sigma, 2)
    V = zeros(Tobs, O, Tobs, O)
    for t1 in 1:Tobs, t2 in 1:Tobs
        lag = abs(t1 - t2)
        if lag >= T
            V[t1, :, t2, :] = zeros(O, O)
        elseif t1 < t2
            V[t1, :, t2, :] = Sigma[lag+1, :, :]
        elseif t1 > t2
            V[t1, :, t2, :] = transpose(Sigma[lag+1, :, :])
        else
            # t1 == t2: diagonal block gets measurement error
            measurement_var  = diagm(sigma_measurement.^2)
            symmetric_sigma  = (Sigma[1, :, :] + transpose(Sigma[1, :, :])) / 2
            V[t1, :, t2, :] = measurement_var + symmetric_sigma
        end
    end
    return reshape(V, Tobs*O, Tobs*O)
end
