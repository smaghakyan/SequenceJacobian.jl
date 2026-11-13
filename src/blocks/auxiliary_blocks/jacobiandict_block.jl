# Manages options and provides methods for computing Jacobians and impulse responses.
# Includes methods for sovling the steady state, impulse response computations, and Jacobian evaluations.

struct JacobianDictBlock <: Block
    M 					# Bijection for tracking variable mappings 
    steady_state_options		# steady state related config
    solve_steady_state_options		# Options for solving the steady state
    impulse_nonlinear_options		# Options for nonlinear impulse response
    solve_impulse_nonlinear_options	# Options for solving nonlinear impulse responses
    impulse_linear_options		# Options for linear impulse response
    jacobian_options			# Options for Jacobian computation
    partial_jacobians_options		# Options for partial Jacobians

    J					# Underlying JacobianDict object
    outputs				# List of outputs
    inputs				# List of inputs
    name				# Name of block
    T					# Time horizon


# Construction of the JacobianDictBlock; intializes internal structures and default option dictionaries.
    function JacobianDictBlock(nesteddict; outputs=nothing, inputs=nothing, name=nothing)
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

        _J = JacobianDict(nesteddict; outputs=outputs, inputs=inputs, name=name)

        return new(_M,
                    _steady_state_options,
                    _solve_steady_state_options,
                    _impulse_nonlinear_options,
                    _solve_impulse_nonlinear_options,
                    _impulse_linear_options,
                    _jacobian_options,
                    _partial_jacobians_options,
                    _J,
                    _J.outputs,
                    _J.inputs,
                    _J.name,
                    _J.T)
    end
end

# Show inputs and outputs to the JacobianDictBlock objects
function Base.show(io::IO, obj::JacobianDictBlock)
    println(io, "<JacobianDictBlock outputs=$(obj.outputs), inputs=$(obj.inputs)>")
end

# compute a linear impulse response using the Jacobians 'Js' and inputs, based on the steady state 'ss'. Returns an 'ImpulseDict'.
function _impulse_linear(obj::JacobianDictBlock, ss, inputs, outputs, Js)
    in_names = OrderedSet{String}(string(k) for k ∈ keys(inputs.toplevel)) ∩ obj.inputs
    J = _jacobian(obj, ss, in_names, outputs; T=inputs.T)
    return apply(J, inputs)[OrderedSet{String}(string(o) for o ∈ outputs)]
end

# Retrieve the Jacobian matrix block for the specified inputs and outputs; throws an error if inputs or outputs are not registered in the block.
function _jacobian(obj::JacobianDictBlock, ss, inputs, outputs; T)
    if !(inputs ⊆ obj.inputs)
        missing_inputs = setdiff(inputs, obj.inputs)
        error("Asking JacobianDictBlock for $missing_inputs, which are among its inputs $(obj.inputs)")
    end
    if !(outputs ⊆ obj.outputs)
        missing_outputs = setdiff(outputs, obj.outputs)
        error("Asking JacobianDictBlock for $missing_outputs, which are among its outputs $(obj.outputs)")
    end
    return obj[(outputs, inputs)]
end
    
# Indexing delegates to the underlying JacobianDict
Base.getindex(b::JacobianDictBlock, k) = Base.getindex(b.J, k)

# A JacobianDictBlock only carries first-order information.
_steady_state(obj::JacobianDictBlock, calibration; kwargs...) =
    error("A JacobianDictBlock ('$(obj.name)') has no steady state: it only holds Jacobians.")
_impulse_nonlinear(obj::JacobianDictBlock, ss, inputs; kwargs...) =
    error("A JacobianDictBlock ('$(obj.name)') cannot produce nonlinear impulse responses: it only holds Jacobians.")
