# drawdag.jl: the DAG of a model as Graphviz DOT source.
# Julia version of Python SSJ utilities/drawdag.py. Python draws the graph with the
# `graphviz` package; here the DOT source is returned (and optionally written to a file
# and rendered with the Graphviz command-line tool `dot`, if it is installed).

"""
    drawdag(model; exogenous=String[], unknowns=String[], targets=String[],
            leftright=false, save=false, savepath=nothing, render=false, format="png")

Graphviz DOT source of the directed acyclic graph of `model` (a `CombinedBlock`, e.g. the
result of `create_model`). Each block is a node, labelled with its name and position in
the topological order; heterogeneous-agent blocks are marked `[HA]`, solved blocks
`[solved]`. An edge from block `j` to block `i` is labelled with the variables `i` takes
from `j`. Exogenous inputs, unknowns and targets, if given, get their own nodes.

* `save=true` writes the source to `savepath * ".dot"` (default `dag/<model name>.dot`).
* `render=true` (with `save=true`) also calls `dot -T<format>` on that file.

Returns the DOT source as a `String`. Paste it into any Graphviz viewer to see the graph.
"""
function drawdag(model; exogenous=String[], unknowns=String[], targets=String[],
                 leftright=false, save=false, savepath=nothing, render=false, format="png")
    hasproperty(model, :blocks) && hasproperty(model, :revadj) ||
        error("drawdag needs a combined block (the result of create_model or combine)")
    exogenous = [string(x) for x in exogenous]
    unknowns  = [string(x) for x in unknowns]
    targets   = [string(x) for x in targets]
    esc(x) = replace(string(x), "\"" => "\\\"")

    lines = String["digraph \"$(esc(model.name))\" {"]
    leftright && push!(lines, "  rankdir=LR;")
    push!(lines, "  ratio=auto; center=true;")
    isempty(exogenous) || push!(lines, "  exog [label=\"exogenous\", shape=box];")
    isempty(unknowns)  || push!(lines, "  unknowns [label=\"unknowns\", shape=box];")
    isempty(targets)   || push!(lines, "  targets [label=\"targets\", shape=diamond];")

    for (i, b) in enumerate(model.blocks)
        kind = (b isa HetBlock || b isa StageBlock) ? " [HA]" : (b isa SolvedBlock ? " [solved]" : "")
        push!(lines, "  b$i [label=\"$(esc(b.name))$kind ($i)\"];")
        inputs = Set(string(k) for k in b.inputs)
        shared = sort([x for x in exogenous if x in inputs])
        isempty(shared) || push!(lines, "  exog -> b$i [label=\"$(join(shared, ", "))\"];")
        shared = sort([x for x in unknowns if x in inputs])
        isempty(shared) || push!(lines, "  unknowns -> b$i [label=\"$(join(shared, ", "))\"];")
        for t in targets
            t in b.outputs && push!(lines, "  b$i -> targets [label=\"$(esc(t))\"];")
        end
        for j in model.revadj[i]
            shared = sort([string(x) for x in b.inputs if x in model.blocks[j].outputs])
            push!(lines, "  b$j -> b$i [label=\"$(join(shared, ", "))\"];")
        end
    end
    push!(lines, "}")
    dot_src = join(lines, "\n")

    if save
        path = isnothing(savepath) ? joinpath("dag", "$(model.name).dot") : string(savepath) * ".dot"
        isempty(dirname(path)) || mkpath(dirname(path))
        open(path, "w") do io
            write(io, dot_src)
        end
        if render
            out = splitext(path)[1] * "." * format
            try
                run(`dot -T$format $path -o $out`)
            catch
                @warn "drawdag: could not run Graphviz `dot`; the DOT source is in $path"
            end
        end
    end
    return dot_src
end
