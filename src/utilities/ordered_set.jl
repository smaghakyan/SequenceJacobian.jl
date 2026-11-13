# OrderedSet wrapper around OrderedCollections.OrderedSet.
# Re-exports with all set operations needed by SSJ.
using OrderedCollections: OrderedSet
# OrderedSet is re-exported via `using OrderedCollections` in the parent module

# Convenience aliases used throughout the codebase
const StrSet = OrderedSet{String}

make_ordered_set(x::Nothing) = nothing
make_ordered_set(x::OrderedSet{String}) = x
make_ordered_set(x::OrderedSet) = OrderedSet{String}(string(k) for k in x)
make_ordered_set(x::AbstractDict) = OrderedSet{String}(string(k) for k in keys(x))
make_ordered_set(x::AbstractString) = OrderedSet{String}([String(x)])
make_ordered_set(x) = OrderedSet{String}(string(k) for k in x)

# Type-coercing set operations: always return OrderedSet{String}
# Single method covering all OrderedSet combinations to avoid ambiguity

# Union: String+String, String+Any, Any+String all → String
function _oset_union_str(a, b)
    seen = OrderedSet{String}()
    for x in a; push!(seen, string(x)); end
    for x in b; push!(seen, string(x)); end
    seen
end
Base.union(a::OrderedSet{String}, b::OrderedSet{String}) = _oset_union_str(a, b)
Base.union(a::OrderedSet{String}, b::OrderedSet)         = _oset_union_str(a, b)
Base.union(a::OrderedSet,         b::OrderedSet{String}) = _oset_union_str(a, b)

function _oset_intersect_str(a, b)
    sb = Set{String}(string(y) for y in b)
    OrderedSet{String}(string(x) for x in a if string(x) in sb)
end
Base.intersect(a::OrderedSet{String}, b::OrderedSet{String}) = _oset_intersect_str(a, b)
Base.intersect(a::OrderedSet{String}, b::OrderedSet)         = _oset_intersect_str(a, b)
Base.intersect(a::OrderedSet,         b::OrderedSet{String}) = _oset_intersect_str(a, b)

function _oset_setdiff_str(a, b)
    sb = Set{String}(string(y) for y in b)
    OrderedSet{String}(string(x) for x in a if string(x) ∉ sb)
end
Base.setdiff(a::OrderedSet{String}, b::OrderedSet{String}) = _oset_setdiff_str(a, b)
Base.setdiff(a::OrderedSet{String}, b::OrderedSet)         = _oset_setdiff_str(a, b)
Base.setdiff(a::OrderedSet,         b::OrderedSet{String}) = _oset_setdiff_str(a, b)
