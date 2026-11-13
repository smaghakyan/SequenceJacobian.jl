# hh_labor: Standard Incomplete Market model with Endogenous Labor Supply.
# Julia port of Python SSJ hetblocks/hh_labor.py

function hh_labor_init(; a_grid, we, r, eis, T)
    # FIX: removed unused `fininc` variable (was computed but never consumed by
    # the backward iteration; caused confusing extra output in the returned tuple)
    coh = max.((1+r) .* reshape(a_grid, 1, :) .+ reshape(we, :, 1) .+ reshape(T, :, 1), 1e-10)
    Va  = (1+r) .* (0.1 .* coh) .^ (-1/eis)
    return (Va=Va,)
end

function _hh_labor(; Va_p, a_grid, we, T, r, beta, eis, frisch, vphi)
    uc_nextgrid = max.(beta .* Va_p, 1e-300)
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, reshape(we, :, 1), eis, frisch, vphi)

    lhs = c_nextgrid .- reshape(we, :, 1) .* n_nextgrid .+ reshape(a_grid, 1, :) .- reshape(T, :, 1)
    rhs = (1+r) .* a_grid

    c = interpolate_y(lhs, reshape(rhs, 1, :), c_nextgrid)
    n = interpolate_y(lhs, reshape(rhs, 1, :), n_nextgrid)

    a = reshape(rhs, 1, :) .+ reshape(we, :, 1) .* n .+ reshape(T, :, 1) .- c
    iconst = findall(a .< a_grid[1])
    a[iconst] .= a_grid[1]
    if !isempty(iconst)
        rows = getindex.(iconst, 1)
        cols = getindex.(iconst, 2)
        c[iconst], n[iconst] = solve_cn(we[rows], rhs[cols] .+ T[rows] .- a_grid[1],
                                         eis, frisch, vphi, Va_p[iconst])
    end

    c   = max.(c, 1e-10)
    Va = (1+r) .* c .^ (-1/eis)
    return (Va=Va, a=a, c=c, n=n)
end

# hh: the HetBlock — matches Python's exported `hh` name
const hh_labor = Base.invokelatest(het, _hh_labor; exogenous="Pi", policy="a", backward="Va", backward_init=hh_labor_init)

# Optimal c, n given marginal utility uc.
function cn(uc, w, eis, frisch, vphi)
    uc_safe = max.(uc, 1e-300)
    return uc_safe .^ (-eis), (w .* uc_safe ./ vphi) .^ frisch
end

function solve_cn(w, T, eis, frisch, vphi, uc_seed)
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)
end

# Solve for uc element-wise (vectorised over inputs).
function solve_uc(w, T, eis, frisch, vphi, uc_seed)
    result = similar(uc_seed)
    for idx in eachindex(uc_seed)
        log_uc = log(uc_seed[idx])
        converged = false
        for _ in 1:30
            ne, ne_p = netexp(log_uc, w[idx], T[idx], eis, frisch, vphi)
            if abs(ne) < 1e-11
                converged = true
                break
            end
            log_uc -= ne / ne_p
        end
        converged || error("Cannot solve constrained household: No convergence after 30 iterations!")
        result[idx] = exp(log_uc)
    end
    return result
end

# Net expenditure and its derivative w.r.t. log uc.
function netexp(log_uc, w, T, eis, frisch, vphi)
    c, n    = cn(exp(log_uc), w, eis, frisch, vphi)
    ne      = c - w * n - T
    c_loguc = -eis * c
    n_loguc = frisch * n
    ne_p    = c_loguc - w * n_loguc
    return ne, ne_p
end
