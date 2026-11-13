using LinearAlgebra

# Simple IdentityMatrix class to initialize Jacobians: cheaper than using the standard identity
struct IdentityMatrix end

# SimpleSparse representation: less efficient individual operations but more general
sparse(i::IdentityMatrix) = SimpleSparse(Dict((0, 0) => 1))

matrix(i::IdentityMatrix, T) = Matrix{Float64}(I, T, T)

Base.show(io::IO, i::IdentityMatrix) = print(io, "IdentityMatrix")

Base.:+(i::IdentityMatrix) = i
Base.:-(i::IdentityMatrix) = -sparse(i)

Base.:*(i::IdentityMatrix, j::AbstractMatrix) = deepcopy(j)
Base.:*(i::AbstractMatrix, j::IdentityMatrix) = deepcopy(i)

#= Efficient representation of sparse linear operators, which are linear combinations of 
basis operators represented by pairs (i, m): where i is index of diagonal on which we have 1s, 
m is number of missing initial entries. Examples: (0,0) is standard identity operator, (-1, 1) has 
1s on diagonal below main diagonal, except in 1st column. 

Jacobian of a SimpleBlock is a SimpleSparse operator with basis elements (i,0). Ideally want more general
(i,m) basis for closure under multiplication. 
=#
mutable struct SimpleSparse
    elements
    indices
    xs

    SimpleSparse(elements) = new(elements, nothing, nothing)
end

for f ∈ (:+, :-, :*)
    @eval Base.$f(i::IdentityMatrix, j::IdentityMatrix) = $f(sparse(i), sparse(j)) 
    @eval Base.$f(i::IdentityMatrix, j::SimpleSparse) = $f(sparse(i), j) 
    @eval Base.$f(i::SimpleSparse, j::IdentityMatrix) = $f(i, sparse(j)) 
    @eval Base.$f(i::IdentityMatrix, j) = $f(sparse(i), j) 
    @eval Base.$f(i, j::IdentityMatrix) = $f(i, sparse(j)) 
end

Base.promote_rule(::Type{SimpleSparse}, ::Type{AbstractArray}) = SimpleSparse

from_simple_diagonals(elements) = SimpleSparse(Dict((i, 0) => x for (i, x) ∈ elements))

matrix(s::SimpleSparse, T) = s + zeros(T, T)

# Rewrite(i,m) dict -> x as pair of arrays, one N*2 array of ints with rows (i,m) and a size N array of floats with entries x
function array!(s::SimpleSparse) 
    if !(s.indices isa Nothing)
        return s.indices, s.xs
    else
        indices, xs = zip(s.elements...)
        s.indices = [indices[i][j] for i ∈ 1:length(indices), j ∈ 1:2]
        s.xs = collect(xs)
        return s.indices, s.xs
    end
end

Base.transpose(s::SimpleSparse) = SimpleSparse(Dict((-i, m) => x for ((i, m), x) ∈ s.elements))

Base.isempty(s::SimpleSparse) = isempty(s.elements)

function nonzero(s::SimpleSparse)
    elements = copy(s.elements)
    for (i, x) ∈ s.elements
        if abs(elements[i]) < 1e-14
            pop!(elements, i)
        end
    end
    return SimpleSparse(elements)
end

iszero(s::SimpleSparse) = isempty(nonzero(s).elements)

Base.:+(s::SimpleSparse) = s
Base.:-(s::SimpleSparse) = SimpleSparse(Dict(i => -x for (i, x) ∈ s.elements))

Base.:*(s::SimpleSparse, t::SimpleSparse) = SimpleSparse(multiply_rs_rs(s, t))
Base.:*(s::SimpleSparse, t::AbstractVector) = multiply_rs_matrix(array!(s)..., reshape(t, (size(t)..., 1)))[:, 1]
Base.:*(s::SimpleSparse, t::AbstractMatrix) = multiply_rs_matrix(array!(s)..., t)
Base.:*(s::AbstractArray, t::SimpleSparse) = transpose(transpose(t) * transpose(s))

# adds SimpleSparse to SimpleSparse by just combining dicts and then summing x's when the bases overlap
function Base.:+(s::SimpleSparse, t::SimpleSparse)
    elements = copy(s.elements)
    for (i, x) ∈ t.elements
        if i ∈ keys(elements)
            elements[i] += x
	    # safeguard to retain sparsity
            if abs(elements[i]) < 1e-14
                pop!(elements, i)
            end
        else
            elements[i] = x
        end
    end
    return SimpleSparse(elements)
end

# adds SimpleSparse to a T*T matrix — matches Python's A.flatten() (row-major) approach
function Base.:+(s::SimpleSparse, t::AbstractMatrix)
    T = size(t)[1]
    t = vec(copy(t)')
    for ((i, m), x) ∈ s.elements
        if i < 0
            t[T * (-i) + (T + 1) * m + 1: T + 1 : end] .+= x
        else
            t[i + (T + 1) * m + 1: T + 1 : (T - i) * T + 1] .+= x
        end
    end
    return reshape(t, (T, T))'
end
Base.:+(s::AbstractMatrix, t::SimpleSparse) = t + s

Base.:-(s::SimpleSparse, t::SimpleSparse) = s + (-t)
Base.:-(s::SimpleSparse, t) = s + (-t)
Base.:-(s, t::SimpleSparse) = s + (-t)


Base.:*(s::SimpleSparse, t::Number) = SimpleSparse(Dict(i => t * x for (i, x) ∈ s.elements))
Base.:*(s::Number, t::SimpleSparse) = t * s

Base.:(==)(s::SimpleSparse, t::SimpleSparse) = s.elements == t.elements

Base.show(io::IO, s::SimpleSparse) = print(io, "SimpleSparse({$(join(["($i, $m) => $x " for ((i, m), x) ∈ s.elements], ", "))})")

# Matrix multiplication of two sparse basis elements: Q(i,m) * Q(j,n) = Q(k,l).
#
# Q(i,m) has ones on diagonal i (i > 0: lead, i < 0: lag) with the first m entries
# of that diagonal missing. This is the rule of Proposition 2 in Auclert et al.
# (2021) in the sign convention of the Python toolbox's `multiply_basis`.
#
# NOTE: an earlier revision replaced this rule with `compute_l` from
# simple_displacement. That function answers a different question (how a
# *time-displaced variable* inside a block is represented) and uses the opposite
# sign convention; as a product rule it is wrong whenever leads and lags are mixed.
# For example lead * lag, Q(1,0)*Q(-1,0), is the identity Q(0,0), but came out as
# Q(0,1), i.e. with the date-0 equation dropped. Checked against dense matrix
# products for all i, j in -3:3 and m, n in 0:3 (see test_models.jl).
function multiply_basis(t1, t2)
    i, m = t1
    j, n = t2
    k = i + j
    if i ≥ 0
        if j ≥ 0
            l = max(m, n - i)
        elseif k ≥ 0
            l = max(m, n - k)
        else
            l = max(m + k, n)
        end
    else
        if j ≤ 0
            l = max(m + j, n)
        else
            l = max(m, n) + min(-i, j)
        end
    end
    return k, l
end

# Matrix multiplication operation on two SimpleSparse objects
function multiply_rs_rs(s1, s2)
    elements = Dict{Tuple{Int,Int},Float64}()
    for (i, x) ∈ s1.elements
        for (j, y) ∈ s2.elements
            k = multiply_basis(i, j)
            if k ∈ keys(elements)
                elements[k] += x * y
            else
                elements[k] = x * y
            end
        end
    end
    return elements
end

# Matrix multiplication of SimpleSparse object and matrix A
function multiply_rs_matrix(indices, xs, A)
    n = size(indices)[1]
    T = size(A)[1]
    S = size(A)[2]
    Aout = zeros(T, S)

    for count ∈ collect(1:n)
        i = indices[count, 1]
        m = indices[count, 2]
        x = xs[count]

        if i == 0
            for t ∈ collect(m+1:T)
                for s ∈ collect(1:S)
                    Aout[t, s] += x * A[t, s]
                end
            end
        elseif i > 0
            for t ∈ collect(m+1:T-i)
                for s ∈ collect(1:S)
                    Aout[t, s] += x * A[t + i, s]
                end
            end
        else
            for t ∈ collect(m-i+1:T)
                for s ∈ collect(1:S)
                    Aout[t, s] += x * A[t + i, s]
                end
            end
        end
    end
    return Aout
end

# call matrix(., T) method if not array (thus SimpleSparse), otherwise standard A
make_matrix(A, T) = matrix(A, T)
make_matrix(A::AbstractArray, T) = A
