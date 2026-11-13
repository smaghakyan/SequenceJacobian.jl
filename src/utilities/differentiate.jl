# Numerical differentiation utilities.
# Julia port of Python SSJ utilities/differentiate.py

# Forward-difference numerical derivative: f'(xss)*shock = (f(xss+h*shock) - f(xss)) / h
function numerical_diff(func, ssinputs_dict::Dict, shock_dict::Dict;
                        h=1e-4, y_ss_list=nothing)
    if isnothing(y_ss_list)
        y_ss_list = make_tuple(func(; ssinputs_dict...))
    end
    shocked = Dict(k => (haskey(shock_dict, k) ?
                         ssinputs_dict[k] + h * shock_dict[k] :
                         ssinputs_dict[k])
                   for k in keys(ssinputs_dict))
    y_list = make_tuple(func(; shocked...))
    return [(y - y_ss) / h for (y, y_ss) in zip(y_list, y_ss_list)]
end

# Central-difference numerical derivative: f'(xss)*shock = (f(xss+h*shock) - f(xss-h*shock)) / 2h
function numerical_diff_symmetric(func, ssinputs_dict::Dict, shock_dict::Dict; h=1e-4)
    up = Dict(k => (haskey(shock_dict, k) ?
                    ssinputs_dict[k] + h * shock_dict[k] :
                    ssinputs_dict[k])
              for k in keys(ssinputs_dict))
    dn = Dict(k => (haskey(shock_dict, k) ?
                    ssinputs_dict[k] - h * shock_dict[k] :
                    ssinputs_dict[k])
              for k in keys(ssinputs_dict))
    y_up = make_tuple(func(; up...))
    y_dn = make_tuple(func(; dn...))
    return [(y_u - y_d) / (2h) for (y_u, y_d) in zip(y_up, y_dn)]
end
