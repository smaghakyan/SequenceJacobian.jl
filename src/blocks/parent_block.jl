abstract type ParentBlock <: Block end

function construct_parent(blocks; name=nothing)
    #if !hasproperty(p, :name) & and !isa(name, Nothing)
    #    p.name = name
    #end
    
    # dict from names to immediate kid blocks themselves
    # dict from descendants to the names of kid blocks through which to access them
    # 'descendants' of a block include itself   
    _name = name
    _kids = Dict{String,Any}()
    _descendants = Dict{String,Any}()

    for block ∈ blocks
        _kids[block.name] = block
        if block isa ParentBlock
            for k ∈ keys(block.descendants)
                if k ∈ keys(_descendants)
                    error("Overlapping block name $k")
                end
                _descendants[k] = block.name
            end
        else
            _descendants[block.name] = block.name
        end
    end

    if name ∈ keys(_descendants)
        error("Overlapping block name '$name': a parent block cannot have the same name as one of the blocks inside it")
    end
    _descendants[name] = nothing

    return _name, _kids, _descendants
end

function Base.getindex(p::ParentBlock, k)
    if k == p.name
        return p
    elseif k ∈ keys(p.kids)
        return p.kids[k]
    else
        return p.kids[p.descendants[k]][k]
    end
end

# If d is a dict with block names as keys and kid is a kid, select only the entries in d that are descendants of a kid
function select(p::ParentBlock, d, kid)
    return Dict(k => v for (k, v) ∈ d if k ∈ keys(p.kids[kid].descendants))
end

function path(p::ParentBlock, k; reverse_flag=true)
    if k ∉ keys(p.descendants) 
        error("Cannot get path to $k since it is not a descendant of the current block")
    end

    if k != p.name
        kid = p.kids[p.descendants[k]]

        if kid isa ParentBlock
            pt = path(kid, k; reverse_flag=false)
        else
            pt = [k]
        end
    else
        pt = []
    end

    push!(pt, p.name)

    if reverse_flag 
        return(reverse(pt))
    else
        return(pt)
    end
end

# Gets attribute attr from descendant k, requiring that attr is a list, dict, set
function get_attribute(p::ParentBlock, k, attr)
    if k == p.name
        inner = getproperty(p, attr)
    else
        kid = p.kids[p.descendants[k]]
        if kid isa ParentBlock
            inner = get_attribute(kid, k, attr)
        else
            inner = getproperty(kid, attr)
            if hasproperty(kid, :M)
                inner = kid.M * inner
            end
        end
    end

    if hasproperty(p, :M)
        return p.M * inner
    else
        return inner
    end
end
