using LinearAlgebra

# Propagates distribution D forward according to 1D policy (x_i, x_pi).
function forward_policy_1d(D, x_i, x_pi)
    nZ, nX = size(D)
    Dnew = zeros(eltype(D), size(D))
    for iz ∈ 1:nZ
        for ix ∈ 1:nX
            i   = x_i[iz, ix]
            _pi = x_pi[iz, ix]
            d   = D[iz, ix]
            Dnew[iz, i]   += d * _pi
            Dnew[iz, i+1] += d * (1 - _pi)
        end
    end
    return Dnew
end

# Computes expected value of X under 1D policy (x_i, x_pi).
function expectation_policy_1d(X, x_i, x_pi)
    nZ, nX = size(X)
    Xnew = zeros(eltype(X), size(X))
    for iz ∈ 1:nZ
        for ix ∈ 1:nX
            i   = x_i[iz, ix]
            _pi = x_pi[iz, ix]
            Xnew[iz, ix] = _pi * X[iz, i] + (1 - _pi) * X[iz, i+1]
        end
    end
    return Xnew
end

# First-order effect of shock to 1D policy on distribution.
function forward_policy_shock_1d(Dss, x_i_ss, x_pi_shock)
    nZ, nX = size(Dss)
    Dshock = zeros(eltype(Dss), size(Dss))
    for iz ∈ 1:nZ
        for ix ∈ 1:nX
            i      = x_i_ss[iz, ix]
            dshock = x_pi_shock[iz, ix] * Dss[iz, ix]
            Dshock[iz, i]   += dshock
            Dshock[iz, i+1] -= dshock
        end
    end
    return Dshock
end

# Propagates distribution D forward according to 2D policy (bilinear interp).
function forward_policy_2d(D, x_i, y_i, x_pi, y_pi)
    nZ, nX, nY = size(D)
    Dnew = zeros(eltype(D), nZ, nX, nY)
    for iz ∈ 1:nZ, ix ∈ 1:nX, iy ∈ 1:nY
        ixp = x_i[iz, ix, iy]
        iyp = y_i[iz, ix, iy]
        β   = x_pi[iz, ix, iy]
        α   = y_pi[iz, ix, iy]
        d   = D[iz, ix, iy]
        Dnew[iz, ixp,   iyp]   += α * β * d
        Dnew[iz, ixp+1, iyp]   += α * (1-β) * d
        Dnew[iz, ixp,   iyp+1] += (1-α) * β * d
        Dnew[iz, ixp+1, iyp+1] += (1-α) * (1-β) * d
    end
    return Dnew
end

# Computes expected value of X under 2D policy (bilinear interp).
function expectation_policy_2d(X, x_i, y_i, x_pi, y_pi)
    nZ, nX, nY = size(X)
    Xnew = zeros(eltype(X), nZ, nX, nY)
    for iz ∈ 1:nZ, ix ∈ 1:nX, iy ∈ 1:nY
        ixp = x_i[iz, ix, iy]
        iyp = y_i[iz, ix, iy]
        α   = x_pi[iz, ix, iy]
        β   = y_pi[iz, ix, iy]
        Xnew[iz, ix, iy] = (α * β         * X[iz, ixp,   iyp]   +
                             α * (1-β)     * X[iz, ixp,   iyp+1] +
                             (1-α) * β     * X[iz, ixp+1, iyp]   +
                             (1-α) * (1-β) * X[iz, ixp+1, iyp+1])
    end
    return Xnew
end

# First-order distributional response to a shock in 2D policy.
function forward_policy_shock_2d(Dss, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock)
    nZ, nX, nY = size(Dss)
    Dshock = zeros(eltype(Dss), nZ, nX, nY)
    for iz ∈ 1:nZ, ix ∈ 1:nX, iy ∈ 1:nY
        ixp = x_i_ss[iz, ix, iy]
        iyp = y_i_ss[iz, ix, iy]
        α   = x_pi_ss[iz, ix, iy]
        β   = y_pi_ss[iz, ix, iy]
        d   = Dss[iz, ix, iy]
        dα  = x_pi_shock[iz, ix, iy] * d
        dβ  = y_pi_shock[iz, ix, iy] * d
        Dshock[iz, ixp,   iyp]   += dα * β + α * dβ
        Dshock[iz, ixp+1, iyp]   += dβ * (1-α) - β * dα
        Dshock[iz, ixp,   iyp+1] += dα * (1-β) - α * dβ
        Dshock[iz, ixp+1, iyp+1] -= (dα * (1-β) + dβ * (1-α))
    end
    return Dshock
end
