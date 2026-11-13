mutable struct CombinedBlock <: ParentBlock
    M
    steady_state_options
    solve_steady_state_options
    impulse_nonlinear_options
    solve_impulse_nonlinear_options
    impulse_linear_options
    jacobian_options
    partial_jacobians_options

    blocks     # Ordered list of sub-blocks
    inmap      # Map of input variables
    outmap     # Map of output variables
    adj        # Forward adjacency/dependency graph
    revadj     # Reverse adjacency graph
    inputs     # list of overall inputs
    outputs    # list of overall outputs

    # Intermediate dependencies and metadata
    _required

    name

    kids
    descendants

    _model_alias

    function CombinedBlock(blocks; name="", model_alias=false, sorted_indices=nothing, intermediate_inputs=nothing)
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

	# ensures all elements are blocks; wrap others in JacobianDictBlock
        blocks_unsorted = [b isa Block ? b : JacobianDictBlock(b) for b ∈ blocks]
	# DAG outputs
        dag = DAG(blocks_unsorted)
        _blocks = dag.blocks
        _inmap = dag.inmap
        _outmap = dag.outmap
        _adj = dag.adj
        _revadj = dag.revadj
        _inputs = dag.inputs
        _outputs = dag.outputs

	# find or use provided intermediate inputs
        __required = intermediate_inputs isa Nothing ? find_intermediate_inputs(_blocks) : intermediate_inputs

	# set default name 
        if name==""
            _name = "$(_blocks[1].name)_to_$(_blocks[end].name)_combined"
        else
            _name = name
        end

        _name, _kids, _descendants = construct_parent(_blocks; name=_name)

        __model_alias = model_alias

        return new(_M,
                    _steady_state_options,
                    _solve_steady_state_options,
                    _impulse_nonlinear_options,
                    _solve_impulse_nonlinear_options,
                    _impulse_linear_options,
                    _jacobian_options,
                    _partial_jacobians_options,
                    _blocks,
                    _inmap,
                    _outmap,
                    _adj,
                    _revadj,
                    _inputs,
                    _outputs,
                    __required,
                    _name,
                    _kids,
                    _descendants,
                    __model_alias)
    end
end

function combine(blocks; name="", model_alias=false)
    return CombinedBlock(blocks; name=name, model_alias=model_alias)
end

function create_model(blocks; name="")
    return combine(blocks; name=name, model_alias=true)
end

function Base.show(io::IO, b::CombinedBlock)
    if b._model_alias
        print(io, "<Model '$(b.name)'>")
    else
        print(io, "<CombinedBlock '$(b.name)'>")
    end
end

# steady state solver: applies each block's steady_state function in topological order 
function _steady_state(b::CombinedBlock, calibration; dissolve, kwargs...)
    ss = copy(calibration)
    for block ∈ b.blocks
        # TODO in python code here re concision of inner_dissolve
        inner_dissolve = [k for k ∈ dissolve if get(b.descendants, k, nothing) == block.name]
        outputs = steady_state(block, ss; dissolve=inner_dissolve, kwargs...)
        merge!(ss, outputs)
    end
    return ss
end

# nonlinear impulse response: propagate the paths through the blocks in DAG order
function _impulse_nonlinear(b::CombinedBlock, ss, inputs; outputs, internals, Js, options, ss_initial)
    original_outputs = outputs
    outputs = setdiff(union(outputs, b._required), _vector_valued(ss))

    impulses = copy(inputs)
    for block ∈ b.blocks
        input_args = Dict{String,Any}(k => v for (k, v) in impulses.toplevel if k in block.inputs)

        # a block none of whose inputs move stays at steady state (unless it starts
        # from a different initial steady state)
        if !isempty(input_args) || !isnothing(ss_initial)
            merge!(impulses, _dyn(impulse_nonlinear, block, ss, ImpulseDict(input_args; T=inputs.T);
                                               outputs = intersect(outputs, block.outputs), internals=internals,
                                               Js=Js, options=options, ss_initial=ss_initial))
        end
    end

    return ImpulseDict(Dict{String,Any}(k => impulses.toplevel[k] for k ∈ original_outputs if k ∈ keys(impulses.toplevel));
                       internals=impulses.internals, T = impulses.T)
end

# linear impulse response: propagate the paths through the blocks in DAG order
function _impulse_linear(b::CombinedBlock, ss, inputs, outputs, Js, options)
    original_outputs = outputs
    outputs = make_ordered_set(setdiff(outputs ∪ b._required, _vector_valued(ss)))

    impulses = copy(inputs)
    for block in b.blocks
        input_args = Dict{String,Any}(k => v for (k, v) in impulses.toplevel if k in block.inputs)
        if !isempty(input_args)
            merge!(impulses, _dyn(impulse_linear, block, ss, ImpulseDict(input_args; T=inputs.T),
                                            intersect(outputs, block.outputs), Js, options))
        end
    end

    return ImpulseDict(Dict{String,Any}(k => impulses.toplevel[k] for k in original_outputs if k in keys(impulses.toplevel));
                       T=inputs.T)
end

# compute partial Jacobians for each block
function _partial_jacobians(b::CombinedBlock, ss, inputs, outputs; T, Js, options)
    vector_valued = _vector_valued(ss)
    inputs  = make_ordered_set(setdiff(inputs  ∪ b._required, vector_valued))
    outputs = make_ordered_set(setdiff(outputs ∪ b._required, vector_valued))

    curlyJs = Dict{String,Any}()
    for block in b.blocks
        curlyJ = partial_jacobians(block, ss, inputs ∩ block.inputs, outputs ∩ block.outputs; T=T, Js=Js, options=options)
        merge!(curlyJs, curlyJ)
    end

    return curlyJs
end

# Compute the full Jacobian using partial derivatives and DAG propagation 
function _jacobian(b::CombinedBlock, ss, inputs, outputs; T, Js, options)
    Js = _partial_jacobians(b, ss, inputs, outputs; T=T, Js=Js, options=options)

    original_outputs = outputs
    total_Js = jacobian_identity(inputs)

    vector_valued = _vector_valued(ss)
    inputs  = make_ordered_set(setdiff(inputs  ∪ b._required, vector_valued))
    outputs = make_ordered_set(setdiff(outputs ∪ b._required, vector_valued))

    for block ∈ b.blocks
        if !isempty(inputs ∩ block.inputs) && !isempty(outputs ∩ block.outputs)
            J = jacobian(block, ss, inputs ∩ block.inputs, outputs ∩ block.outputs; T=T, Js=Js, options=options)
            Jnew = J * total_Js
            if !isempty(Jnew.outputs) && !isempty(Jnew.inputs)
                merge!(total_Js, Jnew)
            end
        end
    end

    return total_Js[original_outputs ∩ total_Js.outputs, :]
end

Model = CombinedBlock
