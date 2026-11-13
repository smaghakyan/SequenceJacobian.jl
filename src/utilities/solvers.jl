# Solvers: Newton, Broyden, and scalar root finders.
# Source: FRBNY SSJ.jl — with multiple bug fixes:
# Those portions: Copyright (c) Federal Reserve Bank of New York, BSD 3-clause license
# (full text and disclaimer in NOTICE.md).
# 1. `dx *x = backtrack_c` -> `dx *= backtrack_c`
# 2. `obtain_j` -> `obtain_J` (wrong capitalisation)
# 3. Missing `end` / stray `else` blocks fixed
# 4. `maximum(abs(y))` -> `maximum(abs, y)`
# 5. Added brentq from Roots.jl

using LinearAlgebra
using Printf
using Roots: find_zero, Brent

function newton_solver(f, x0, y0=nothing; tol=1e-9, maxcount=100, backtrack_c=0.5, verbose=true)
    x = copy(x0)
    y = isnothing(y0) ? f(x) : copy(y0)

    for count in 1:maxcount
        verbose && printit(count, x, y)
        maximum(abs, y) < tol && return x, y

        J  = obtain_J(f, x, y)
        dx = J \ (-y)

        success = false
        for bcount in 1:30
            try
                ynew = f(x + dx)
                predicted = -sum((J * dx) .* y) * ((1 - 1/2^bcount) + 1) / 2
                actual    = (sum(y.^2) - sum(ynew.^2)) / 2
                if actual < predicted / 2
                    verbose && println("Backtracking")
                    dx .*= backtrack_c   # FIX: was `dx *x = backtrack_c`
                else
                    y  = ynew
                    x += dx
                    success = true
                    break
                end
            catch e
                verbose && println("Backtracking")
                dx .*= backtrack_c
            end
        end
        success || error("Too many backtracks, maybe bad initial guess?")
    end
    error("No convergence after $maxcount iterations")
end

function broyden_solver(f, x0, y0=nothing; tol=1e-9, maxcount=100, backtrack_c=0.5, verbose=true)
    x = copy(x0)
    y = isnothing(y0) ? f(x) : copy(y0)
    J = nothing

    for count in 1:maxcount
        verbose && @printf "Iteration: %d\n" count
        maximum(abs, y) < tol && return x, y

        count == 1 && (J = obtain_J(f, x, y))   # FIX: was `obtain_j`

        nx, ny = length(x), length(y)
        dx = if nx == ny
            J \ (-y)
        elseif nx < ny
            @warn "dim(x)=$nx < dim(y)=$ny, using least-squares"
            J \ (-y)
        else
            error("dim(x)=$nx > dim(y)=$ny, underdetermined")
        end

        success = false
        for _ in 1:30
            try
                ynew = f(x + dx)
                J = broyden_update(J, dx, ynew - y)
                y  = ynew
                x += dx
                success = true
                break
            catch
                verbose && println("Backtracking")
                dx .*= backtrack_c
            end
        end
        success || error("Too many backtracks")
    end
    error("No convergence after $maxcount iterations")
end

function obtain_J(f, x, y; h=1e-5)
    nx, ny = length(x), length(y)
    J = Matrix{Float64}(undef, ny, nx)
    for i in 1:nx
        dx = zeros(nx); dx[i] = h
        J[:,i] = (f(x .+ dx) .- y) / h
    end
    return J
end

function broyden_update(J, dx, dy)
    return J + ((dy - J*dx) / norm(dx)^2) * dx'
end

function printit(it, x, y)
    println("On iteration $it")
    println("x = " * join([@sprintf("%.3f", xi) for xi in x], ","))
    println("y = " * join([@sprintf("%.3f", yi) for yi in y], ","))
    println()
end

# Brent's method for scalar root finding — wraps Roots.jl.
function brentq(f, a::Real, b::Real; tol=1e-12, maxiter=100)
    return find_zero(f, (a, b), Brent(); atol=tol, maxevals=maxiter)
end
