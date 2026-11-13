# Struct definition of a simple block

mutable struct SimpleBlock <: Block
    M
    steady_state_options
    solve_steady_state_options
    impulse_nonlinear_options
    solve_impulse_nonlinear_options
    impulse_linear_options
    jacobian_options
    partial_jacobians_options

    f
    name
    inputs
    outputs

    function SimpleBlock(f::Function)
        _M = Bijection(Dict())
        _steady_state_options = Dict()
        _solve_steady_state_options = Dict{String, Any}(
            "solver" => "",
            "solver_kwargs" => Dict(),
            "ttol" => 1e-12,
            "ctol" => 1e-9,
            "verbose" => false,
            "constrained_method" => "linear_continuation",
            "constrained_kwargs" => Dict())
        _impulse_nonlinear_options = Dict()
        _solve_impulse_nonlinear_options = Dict{String, Any}(
            "tol" => 1e-8,
            "maxit" => 30,
            "verbose" => true,
            "backtrack" => true,
            "max_backtracks" => 25,
            "warm_start" => false)
        _impulse_linear_options = Dict()
        _jacobian_options = Dict()
        _partial_jacobians_options = Dict()

        f_ext = ExtendedFunction(f)

        return new(_M,
                    _steady_state_options,
                    _solve_steady_state_options,
                    _impulse_nonlinear_options,
                    _solve_impulse_nonlinear_options,
                    _impulse_linear_options,
                    _jacobian_options,
                    _partial_jacobians_options,
                    f_ext,
                    f_ext.name,
                    f_ext.inputs,
                    f_ext.outputs)
    end
end

function simple(f)
    return SimpleBlock(f)
end

Base.show(io::IO, b::SimpleBlock) = print(io, "<SimpleBlock '$(b.name)'>")

function _steady_state(b::SimpleBlock, ss)
    outputs = wrapped_call(b.f, ss; preprocess=ignore, postprocess=numeric_primitive)
    return SteadyStateDict(Dict(ss..., outputs...))
end

function _impulse_nonlinear(b::SimpleBlock, ss, inputs; outputs, ss_initial)
    if isnothing(ss_initial)
        ss_initial = ss
        ss_initial_flag = false
    else
        ss_initial_flag = true
    end

    input_args = Dict{String,Any}()
    for (k, v) ∈ inputs
        if v isa Real
            error("Keyword argument $k = $v is scalar, should be time path.")
        end
        input_args[string(k)] = Displace(v .+ ss[string(k)]; ss=ss[k], ss_initial=ss_initial[k], name=k)
    end

    for k ∈ b.f.inputs          # unmapped names (b.inputs may be remapped)
        if k ∉ keys(input_args)
            if !ss_initial_flag || (ss_initial_flag && ss_initial[k] == ss[k])
                input_args[k] = ignore(ss[k])
            else
                input_args[k] = Displace(fill(ss[k], inputs.T); ss=ss[k], ss_initial=ss_initial[k], name=k)
            end
        end
    end
    return ImpulseDict(make_impulse_uniform_length(b.f(input_args), inputs.T); T=inputs.T)[outputs] - ss
end

# Partial-equilibrium linear impulse: apply the block's (analytic) Jacobian.
function _impulse_linear(b::SimpleBlock, ss, inputs, outputs, Js)
    T = inputs.T
    in_names = OrderedSet{String}(string(k) for k ∈ keys(inputs.toplevel)) ∩ b.f.inputs
    J = _jacobian(b, ss, in_names, outputs; T=T)
    return apply(J, inputs)[OrderedSet{String}(string(o) for o ∈ outputs)]
end

function _jacobian(b::SimpleBlock, ss, inputs, outputs; T)
    # invertedJ[i_str] = Dict{String,Any}(output => SimpleSparse or empty Dict)
    invertedJ = Dict{String, Dict{String,Any}}()
    for i ∈ inputs
        invertedJ[string(i)] = compute_single_shock_J(b, ss, i)
    end

    # J[o_str][i_str] = SimpleSparse (only non-zero entries)
    J = Dict{String, Dict{String,Any}}(string(o) => Dict{String,Any}() for o ∈ outputs)
    for o ∈ outputs
        o_str = string(o)
        for i ∈ inputs
            i_str = string(i)
            v = get(invertedJ[i_str], o_str, nothing)
            if !isnothing(v) && v isa SimpleSparse && !iszero(v)
                J[o_str][i_str] = v
            end
        end
    end

    return JacobianDict(J; outputs=outputs, inputs=inputs, name=b.name, T=T)
end

function compute_single_shock_J(b::SimpleBlock, ss, i)
    i_str = string(i)
    input_args = Dict{String, Union{Ignore, AccumulatedDerivative}}(
        string(k) => ignore(ss[string(k)]) for k ∈ b.inputs)
    input_args[i_str] = AccumulatedDerivative(f_value=numeric_primitive(ss[i_str]))

    # Returns Dict{String, Any} mapping output_name => SimpleSparse (for this one input i)
    # Empty dict means zero Jacobian for that output
    J = Dict{String, Any}(string(o) => Dict{String,Any}() for o ∈ b.outputs)
    for (o_name, o) in b.f(input_args)
        if o isa AccumulatedDerivative
            J[string(o_name)] = SimpleSparse(o.elements)
        end
    end

    return J
end

# Outputs that do not depend on any time-varying input come back as scalars: expand
# them to constant paths of length T.
function make_impulse_uniform_length(out, T=nothing)
    isnothing(T) && (T = maximum([length(v) for v in values(out)]))
    return Dict{String,Any}(string(k) => (v isa Real ? fill(numeric_primitive(v), T) : collect(Float64, numeric_primitive(v))) for (k, v) in out)
end
