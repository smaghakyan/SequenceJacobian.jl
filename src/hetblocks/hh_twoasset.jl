# hh_twoasset: Two-Asset HANK household block.
# Julia port of Python SSJ hetblocks/hh_twoasset.py

function hh_twoasset_init(; b_grid, a_grid, z_grid, eis)
    nZ = size(z_grid, 1)
    Va_temp = (0.6 .+ 1.1 .* b_grid .+ a_grid') .^ (-1 / eis)
    Va = reshape(Va_temp, 1, size(Va_temp)...) .* ones(nZ, 1, 1)
    Vb_temp = (0.5 .+ b_grid .+ 1.2 .* a_grid') .^ (-1 / eis)
    Vb = reshape(Vb_temp, 1, size(Vb_temp)...) .* ones(nZ, 1, 1)
    return (Va=Va, Vb=Vb)
end

# Adjustment costs hetoutput.
function adjustment_costs(; a, a_grid, ra, chi0, chi1, chi2)
    # a is (nZ,nB,nA) — reshape a_grid to broadcast correctly
    a_grid_r = ndims(a) == 3 ? reshape(a_grid, 1, 1, :) : a_grid
    chi = get_Psi_and_deriv(a, a_grid_r, ra, chi0, chi1, chi2)[1]
    return (chi=chi,)   # lowercase — aggregated to uppercase CHI by HetBlock
end

# Precompute marginal cost grid (hetinput).
function marginal_cost_grid(; a_grid, ra, chi0, chi1, chi2)
    Psi1 = get_Psi_and_deriv(reshape(a_grid, :, 1), reshape(a_grid, 1, :),
                              ra, chi0, chi1, chi2)[2]
    return (Psi1=Psi1,)
end

function _hh_twoasset(; Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid,
               beta, eis, rb, ra, chi0, chi1, chi2, Psi1)
    Wb = beta .* Vb_p
    Wa = beta .* Va_p
    W_ratio = Wa ./ Wb

    # STEP 3: UNCONSTRAINED a' — interpolate where W_ratio == 1 + Psi1
    i, pi_w = lhs_equals_rhs_interpolate(W_ratio, 1 .+ Psi1)
    a_endo_unc = apply_coord(i, pi_w, a_grid)
    c_endo_unc = apply_coord(i, pi_w, Wb)
    c_endo_unc[c_endo_unc .< 0] .= zero(eltype(c_endo_unc))
    c_endo_unc = c_endo_unc .^ (-eis)

    # STEP 4: UNCONSTRAINED b'(z,b,a), a'(z,b,a)
    # FIX: budget uses rb (deposit rate) when b'>0, rl (loan rate) when b'<0.
    # Standard model has rb==rl so this is numerically identical; but it is
    # required for correctness when rb ≠ rl (models with a spread between the two rates).
    _num_unc = (c_endo_unc .+ a_endo_unc .+
              addouter(-z_grid, b_grid, -(1 .+ ra) .* a_grid) .+
              get_Psi_and_deriv(a_endo_unc, reshape(a_grid, 1, 1, :), ra, chi0, chi1, chi2)[1])
    b_endo = _num_unc ./ (1 .+ rb)   # rb is scalar in standard model; no change needed here
    i, pi_w = interpolate_coord(permutedims(b_endo, (1, 3, 2)), b_grid)
    a_unc = permutedims(apply_coord(i, pi_w, permutedims(a_endo_unc, (1, 3, 2))), (1, 3, 2))
    b_unc = permutedims(apply_coord(i, pi_w, b_grid), (1, 3, 2))

    # STEP 5: CONSTRAINED a'(z,kappa,a) — vectorized over all (z,kappa,a) simultaneously
    lhs_con = W_ratio[:, 1:1, :] ./ (1 .+ reshape(k_grid, 1, :, 1))
    i, pi_w = lhs_equals_rhs_interpolate(lhs_con, 1 .+ Psi1)
    a_endo_con = apply_coord(i, pi_w, a_grid)
    c_endo_con = apply_coord(i, pi_w, Wb[:, 1:1, :])
    c_endo_con[c_endo_con .< 0] .= zero(eltype(c_endo_con))
    c_endo_con = (1 .+ reshape(k_grid, 1, :, 1)) .^ (-eis) .* c_endo_con .^ (-eis)

    # STEP 6: CONSTRAINED a'(z,b,a) via kappa→b interpolation
    # FIX: constrained branch also uses rb (b' = b_grid[1] = borrowing constraint)
    _num_con = (c_endo_con .+ a_endo_con .+
                  addouter(-z_grid, fill(b_grid[1], length(k_grid)), -(1 .+ ra) .* a_grid) .+
                  get_Psi_and_deriv(a_endo_con, reshape(a_grid, 1, 1, :), ra, chi0, chi1, chi2)[1])
    b_endo_con = _num_con ./ (1 .+ rb)
    a_con = permutedims(interpolate_y(permutedims(b_endo_con, (1, 3, 2)), b_grid,
                                      permutedims(a_endo_con, (1, 3, 2))), (1, 3, 2))

    # STEP 7: Combine unconstrained and constrained solutions
    a, b = copy(a_unc), copy(b_unc)
    b[b .<= b_grid[1]] .= b_grid[1]
    a[b .<= b_grid[1]] .= a_con[b .<= b_grid[1]]

    Psi, _, Psi2 = get_Psi_and_deriv(a, reshape(a_grid, 1, 1, :), ra, chi0, chi1, chi2)
    c = addouter(z_grid, (1 .+ rb) .* b_grid, (1 .+ ra) .* a_grid) .- Psi .- a .- b
    c = max.(c, 1e-10)   # floor to avoid NaN in uc
    uc = c .^ (-1 / eis)
    uce = reshape(e_grid, :, 1, 1) .* uc
    Va = (1 .+ ra .- Psi2) .* uc
    Vb = (1 .+ rb) .* uc
    return (Va=Va, Vb=Vb, a=a, b=b, c=c, uce=uce)
end

# hh: the HetBlock — matches Python's exported `hh` name
const hh_twoasset = Base.invokelatest(
    het, _hh_twoasset;
    exogenous="Pi", policy=["b","a"], backward=["Vb","Va"],
    hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs],
    backward_init=hh_twoasset_init)

# Supporting functions

# Take matrix A times X[:, i1, i2, ...] for each i1, i2, ...
# Same as A @ X when X is 1D or 2D.
function matrix_times_first_dim(A, X)
    return reshape(A * reshape(X, size(X, 1), :), size(A, 1), size(X)[2:end]...)
end

# Adjustment cost Psi(ap,a) and its derivatives.
function get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2)
    a_with_return   = (1+ra) .* a
    a_change        = ap .- a_with_return
    abs_a_change    = abs.(a_change)
    sign_change     = sign.(a_change)
    adj_denominator = a_with_return .+ chi0
    core_factor     = (abs_a_change ./ adj_denominator) .^ (chi2 - 1)
    Psi  = chi1/chi2 .* abs_a_change .* core_factor
    Psi1 = chi1 .* sign_change .* core_factor
    Psi2 = -(1+ra) .* (Psi1 .+ (chi2-1) .* Psi ./ adj_denominator)
    return Psi, Psi1, Psi2
end

# Outer sum: result[i,j,k] = z[i] + b[j] + a[k]
function addouter(z, b, a)
    return reshape(z,:,1,1) .+ reshape(b,1,:,1) .+ reshape(a,1,1,:)
end

# For each j, find i where lhs[i] crosses rhs[i,j] (1D core).
function lhs_equals_rhs_interpolate(lhs::AbstractVector, rhs::AbstractMatrix)
    ni, nj = size(rhs)
    iout  = zeros(Int32, nj)
    piout = zeros(Float64, nj)
    i = 1
    for j in 1:nj
        while true
            lhs[i] < rhs[i,j] && break
            i < ni ? (i += 1) : break
        end
        if i == 1
            iout[j] = 1; piout[j] = 1.0
        else
            iout[j] = i - 1
            eu = rhs[i,j] - lhs[i]; el = rhs[i-1,j] - lhs[i-1]
            piout[j] = eu / (eu - el)
        end
    end
    return iout, piout
end

# Broadcast lhs_equals_rhs_interpolate over higher dims.
function lhs_equals_rhs_interpolate(lhs, rhs)
    nj = size(rhs)[end]
    ndims(lhs)==1 && (lhs = reshape(lhs,(1,1,:)))
    ndims(lhs)==2 && (lhs = reshape(lhs,(1,size(lhs)...)))
    ndims(rhs)==2 && (rhs = reshape(rhs,(1,size(rhs)...)))
    d1 = size(lhs,1); d2 = max(size(lhs,2), size(rhs,1))
    size(lhs,2)!=d2 && (lhs = repeat(lhs,1,d2,1))
    size(rhs,1)!=d2 && (rhs = repeat(rhs,d2,1,1))
    iout  = zeros(Int32,   d1, d2, nj)
    piout = zeros(Float64, d1, d2, nj)
    for i1 in 1:d1, i2 in 1:d2
        iout[i1,i2,:], piout[i1,i2,:] = lhs_equals_rhs_interpolate(lhs[i1,i2,:], rhs[i2,:,:])
    end
    (d1==1 && d2==1) && return iout[1,1,:], piout[1,1,:]
    d1==1 && return iout[1,:,:], piout[1,:,:]
    d2==1 && return iout[:,1,:], piout[:,1,:]
    return iout, piout
end
