# convenience.jl — High-level convenience wrappers matching Python SSJ public API
# Mirrors sj.block.solve_impulse_linear, sj.block.solve_jacobian, etc.

# Turn scalars / short vectors into paths of length T.
function _shock_paths(inputs::AbstractDict, T::Int)
    shock_paths = Dict{String, Any}()
    for (k, v) in inputs
        if v isa Number
            shock_paths[string(k)] = fill(Float64(v), T)
        elseif length(v) < T
            tmp = zeros(T); tmp[1:length(v)] .= v; shock_paths[string(k)] = tmp
        else
            shock_paths[string(k)] = Float64.(v[1:T])
        end
    end
    return ImpulseDict(shock_paths; T=T)
end

"""
    impulse_linear(model, ss, inputs, unknowns, targets, exog_names; T=300, Js=Dict(), outputs=nothing)

Compute GE linear impulse responses. Returns `(irfs::Dict, G::JacobianDict)`.

`inputs` is a Dict of shock name => path (a scalar is broadcast to T periods, a
short vector is padded with zeros). `irfs` contains the shocks, the unknowns and
every scalar model output.

# Example
```julia
dr = -0.0025 * 0.9.^(0:T-1)
irfs, G = impulse_linear(model, ss, Dict("rstar" => dr),
                         unknowns, targets, exog_names; T=300)
dY = irfs["Y"]   # output impulse response
```
"""
function impulse_linear(model, ss, inputs::Dict, unknowns, targets, exog_names;
                        T::Int=300, Js::Dict=Dict(), outputs=nothing)
    shock_dict = _shock_paths(inputs, T)

    result = solve_impulse_linear(model, ss,
                                  OrderedSet{String}(string(k) for k in unknowns),
                                  OrderedSet{String}(string(k) for k in targets),
                                  shock_dict,
                                  isnothing(outputs) ? nothing : OrderedSet{String}(string(k) for k in outputs);
                                  Js=Js)

    # Also compute the GE Jacobian for convenience
    G = solve_jacobian(model, ss,
                       OrderedSet{String}(string(k) for k in unknowns),
                       OrderedSet{String}(string(k) for k in targets),
                       OrderedSet{String}(string(k) for k in exog_names);
                       T=T, Js=Js)

    irfs = Dict{String, Vector{Float64}}(k => collect(Float64, v) for (k, v) in result.toplevel)
    return irfs, G
end

"""
    td_nonlinear(model, ss, inputs, unknowns, targets, exog_names; T=300, maxit=30, tol=1e-8, ...)

Nonlinear perfect-foresight transition after an unanticipated ("MIT") shock,
solved by quasi-Newton iteration on the steady-state Jacobian with a backtracking
step. Returns an ImpulseDict of deviations from steady state.

Keywords: `verbose`, `backtrack`, `warm_start` (start from the linear solution),
`internals` (e.g. `Dict("hh" => ["D", "Dbeg_terminal"])` to keep the distribution path),
`Js`, `H_U_factored`.

# Example
```julia
shock = Dict("rstar" => -0.01 * 0.9.^(0:T-1))
result = td_nonlinear(model, ss, shock, unknowns, targets, exog_names; T=300)
dY = result["Y"]
```
"""
function td_nonlinear(model, ss, inputs::Dict, unknowns, targets, exog_names=nothing;
                      T::Int=300, maxit::Int=30, tol::Float64=1e-8,
                      verbose::Bool=true, backtrack::Bool=true, warm_start::Bool=false,
                      internals=Dict(), Js::Dict=Dict(), H_U_factored=nothing)
    shock_dict = _shock_paths(inputs, T)

    # (tol/maxit/verbose used to be wrapped in a dictionary keyed by the method name,
    #  where no block ever looked for them, so they were silently ignored)
    return solve_impulse_nonlinear(model, ss,
                                   OrderedSet{String}(string(k) for k in unknowns),
                                   OrderedSet{String}(string(k) for k in targets),
                                   shock_dict;
                                   internals=internals, Js=Js, H_U_factored=H_U_factored,
                                   tol=tol, maxit=maxit, verbose=verbose,
                                   backtrack=backtrack, warm_start=warm_start)
end

"""
    build_MA_representation(G, observables, shocks, T)

Stack the impulse responses of `observables` to unit date-0 innovations in `shocks`
into the `(T, O, Z)` array expected by `all_covariances`.

`G[obs][shock][:, 1]` is the response to a one-off unit shock at date 0. For an
AR(1) shock with persistence ρ, pass the response to the whole path instead, e.g.
`G[obs][shock] * ρ.^(0:T-1)`.

# Example
```julia
G = solve_jacobian(model, ss, unknowns, targets, exog_names; T=300)
M = build_MA_representation(G, ["Y", "pi", "r"], ["rstar"], 300)
Sigma = all_covariances(M, [1.0])
```
"""
function build_MA_representation(G, observables, shocks, T)
    O = length(observables); Z = length(shocks)
    M = zeros(T, O, Z)
    for (iz, shock) in enumerate(shocks)
        for (io, obs) in enumerate(observables)
            if obs in keys(G) && shock in keys(G[obs])
                J = make_matrix(G[obs][shock], T)
                # column 1 is the IRF to a unit shock at t=0
                M[:, io, iz] .= J[:, 1]
            end
        end
    end
    return M
end

"""
    get_shock_path(T; rho=0.9, scale=-0.0025)

AR(1) monetary policy shock path of length T.
"""
get_shock_path(T; rho=0.9, scale=-0.0025) = scale .* rho .^ (0:T-1)

"""
    irf_pct(G, var, shock, dr; ss_val=1.0)

Extract percentage IRF: 100 * G[var][shock] * dr / ss_val
"""
function irf_pct(G, var, shock, dr; ss_val=1.0)
    haskey(G, var) && haskey(G[var], shock) ? 100 .* (G[var][shock] * dr) ./ ss_val : zeros(length(dr))
end
