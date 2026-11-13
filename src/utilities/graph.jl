# Graph: DAG and topological sort.
# Source: FRBNY SSJ.jl (clean, no bugs found)
# Those portions: Copyright (c) Federal Reserve Bank of New York, BSD 3-clause license
# (full text and disclaimer in NOTICE.md).

using OrderedCollections: OrderedSet

struct DAG
    blocks
    inmap
    outmap
    adj
    revadj
    inputs
    outputs

    function DAG(blocks)
        _inmap  = get_input_map(blocks)
        _outmap = get_output_map(blocks)
        _adj    = get_block_adjacency_list(blocks, _inmap)
        _revadj = get_block_reverse_adjacency_list(blocks, _outmap)
        topsort = topological_sort(_adj, _revadj; names=[block.name for block in blocks])

        M = Bijection(Dict(t => i for (i,t) in enumerate(topsort)))

        return new(
            [blocks[t] for t in topsort],
            Dict(k => (M * v) for (k,v) in _inmap),
            Dict(k => (M * v) for (k,v) in _outmap),
            [M * _adj[t]    for t in topsort],
            [M * _revadj[t] for t in topsort],
            OrderedSet{String}(k for k in keys(_inmap) if k ∉ keys(_outmap)),
            OrderedSet{String}(keys(_outmap))
        )
    end
end

function get_input_map(blocks)
    inmap = Dict{String, OrderedSet{Int}}()
    for (n, block) in enumerate(blocks)
        for i in block.inputs
            inset = get!(inmap, i, OrderedSet{Int}())
            push!(inset, n)
        end
    end
    return inmap
end

function get_output_map(blocks)
    outmap = Dict{String, Int}()
    for (n, block) in enumerate(blocks)
        for o in block.outputs
            if o in keys(outmap)
                other_n = outmap[o]
                error("\"$o\" is output of both block $n ($(blocks[n].name)) and block $other_n ($(blocks[other_n].name))")
            end
            outmap[o] = n
        end
    end
    return outmap
end

function get_block_adjacency_list(blocks, inmap)
    adj = Vector{OrderedSet{Int}}()
    for block in blocks
        cur = OrderedSet{Int}()
        for o in block.outputs
            o in keys(inmap) && (cur = cur ∪ inmap[o])
        end
        push!(adj, cur)
    end
    return adj
end

function get_block_reverse_adjacency_list(blocks, outmap)
    revadj = Vector{OrderedSet{Int}}()
    for block in blocks
        cur = OrderedSet{Int}()
        for i in block.inputs
            i in keys(outmap) && push!(cur, outmap[i])
        end
        push!(revadj, cur)
    end
    return revadj
end

function topological_sort(adj, revadj; names=nothing)
    dep    = [copy(s) for s in revadj]
    revdep = adj
    nodeps = [n for (n, ds) in enumerate(dep) if isempty(ds)]
    topsorted = Int[]

    while !isempty(nodeps)
        n = pop!(nodeps)
        push!(topsorted, n)
        for n2 in revdep[n]
            delete!(dep[n2], n)
            isempty(dep[n2]) && push!(nodeps, n2)
        end
    end

    if length(topsorted) != length(dep)
        cycle = find_cycle(dep, setdiff(Set(1:length(dep)), Set(topsorted)))
        @assert !isnothing(cycle)
        cycle_names = isnothing(names) ? cycle : [names[i] for i in cycle]
        error("Topological sort failed: cyclic dependency $(join(string.(cycle_names), " -> "))")
    end
    return topsorted
end

function find_cycle(dep, onlyset)
    dep = Dict{Int, OrderedSet{Int}}(k => (dep[k] ∩ onlyset) for k in collect(1:length(dep)) if k in onlyset)
    tovisit = Set{Int}(keys(dep))
    stack   = OrderedSet{Int}()

    while !isempty(tovisit) || !isempty(stack)
        if !isempty(stack)
            n = collect(stack)[end]
            if !isempty(dep[n])
                n2 = pop!(dep[n])
                if n2 in stack
                    i2loc = findfirst(==(n2), collect(stack))
                    return vcat(collect(stack)[i2loc:end], [collect(stack)[i2loc]])
                elseif n2 in tovisit
                    delete!(tovisit, n2)
                    push!(stack, n2)
                end
            else
                delete!(stack, n)
            end
        else
            n = pop!(tovisit)
            push!(stack, n)
        end
    end
    return nothing
end

function visit_from_inputs(d::DAG, inputs)
    inputs = (inputs isa Dict ? OrderedSet{String}(keys(inputs)) : OrderedSet{String}(string(k) for k in inputs)) ∩ d.inputs
    visited = OrderedSet{Int}()
    for (n, (block, parentset)) in enumerate(zip(d.blocks, d.revadj))
        found = any(i in block.inputs for i in inputs)
        if found
            push!(visited, n)
        elseif !isempty(visited ∩ parentset)
            push!(visited, n)
        end
    end
    return visited
end

function visit_from_outputs(d::DAG, outputs)
    outputs = OrderedSet{String}(string(k) for k in outputs) ∩ d.outputs
    visited = OrderedSet{Int}()
    for n in reverse(1:length(d.blocks))
        block    = d.blocks[n]
        childset = d.adj[n]
        found = any(o in block.outputs for o in outputs)
        if found
            push!(visited, n)
        elseif !isempty(visited ∩ childset)
            push!(visited, n)
        end
    end
    return OrderedSet{Int}(reverse(collect(visited)))
end

function find_intermediate_inputs(blocks)
    required = OrderedSet{String}()
    outmap   = get_output_map(blocks)
    for block in blocks
        inputs = hasproperty(block, :inputs) ? block.inputs : OrderedSet{String}()
        for i in inputs
            i in keys(outmap) && push!(required, string(i))
        end
    end
    return required
end
