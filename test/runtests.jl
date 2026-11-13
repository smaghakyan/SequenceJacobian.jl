# runtests.jl: regression tests for SequenceJacobian.jl
#
#   julia --project -e 'using Pkg; Pkg.test()'      (from the repository root)
#   julia --project test/runtests.jl                 (same, without Pkg)
#
# Reference numbers come from the official Python toolbox (sequence-jacobian 1.0.0)
# run on the same example models (rbc.dag(), krusell_smith.dag(), hank.dag()).
# Tolerances are loose enough for different root finders and BLAS libraries and tight
# enough to catch a wrong Jacobian (an error in the product of lead and lag operators
# moves the RBC impact response of output from 0.015901 to 0.016252).
#
# The labor-choice model (a StageBlock) has no Python counterpart; its tests check
# internal consistency: market clearing, Walras' law, the fake-news Jacobian against a
# brute-force one, and linear against nonlinear transitions for a small shock.

using Test, LinearAlgebra
using SequenceJacobian
const SJ = SequenceJacobian

close_to(a, b; tol=1e-7) = maximum(abs.(collect(a)[1:length(b)] .- b)) < tol

# ---- reference numbers (Python sequence-jacobian 1.0.0) -------------------------
const RBC_K = 3.1428571428571357; const RBC_Z = 0.8816460975214568
const RBC_dY = [0.01590111098771017, 0.014230591353967464, 0.01273626129195053, 0.011399463448669567, 0.010203524209507585, 0.009133541914539642, 0.008176197777111743, 0.007319587061278925]
const RBC_dK = [0.010688717879671895, 0.019093900406138363, 0.025581935216969286, 0.030466885020747122, 0.03401750266235619, 0.03646334211060611, 0.03800007879609535, 0.038794138768427376]
const RBC_dC = [0.005212393108038267, 0.005558190880509194, 0.005770878970966136, 0.005874965264467498, 0.005891234442379828, 0.00583726489973082, 0.005727877538857345, 0.005575525119044513]
const RBC_nlY = [0.015925120342750176, 0.014252638557207264, 0.012755856988217129, 0.0114164520654636, 0.01021795652040458, 0.009145588876809274, 0.008186093091091085, 0.007327589661313416]
const RBC_nlK = [0.010713628456996787, 0.019139913792992222, 0.025645929693643037, 0.03054611073143287, 0.03410942273345895, 0.03656556583974058, 0.03811034581833963, 0.03891032823965235]
const RBC_nl10Y = [0.16130949170357378, 0.14446135889239797, 0.12929539329705642, 0.11567615260312158, 0.1034655350496636, 0.09252989372807097, 0.08274353648725952, 0.07399025568995254]
const KS_beta = 0.9813743595924695
const KS_J_A_r_11 = 3.0375292243243326; const KS_J_A_r_64 = 4.497346209757661; const KS_J_C_w_38 = 0.023826195487637734
const KS_dK = [0.006211964474973497, 0.011282020199349792, 0.015368488475405175, 0.018609986603550274, 0.021127792650609524, 0.023027903800252537, 0.024402847401197122, 0.025333280377597394]
const KS_dC = [0.005130455152764498, 0.0052002415851195504, 0.005213711824009807, 0.005180810664690411, 0.005110055336081794, 0.00500875214210436, 0.004883162265634199, 0.004738639643201219]
const KS_peA = [0.005163586184164348, 0.009167840010295247, 0.012245992054555455, 0.0145851902985763, 0.016335491976087635, 0.017617133245630523]
const KS_peC = [-0.002020729037762181, -0.0014342047894047028, -0.000969181063496638, -0.0006013289267957633, -0.00031116480327875173, -8.308551562685729e-05]
const KS_pelinA = [0.005166944551255076, 0.009168969332124554, 0.012242346701370574, 0.014575947219041546, 0.016320753543710223, 0.017597455739670986]
const KS_nlK = [0.006214227288693222, 0.011287681512981623, 0.015378099231195545, 0.018623685708164462, 0.02114543894076427, 0.023049174203160556, 0.0244273094661511, 0.025360445200725362]
const KS_nlC = [0.005128192359934891, 0.005198889493316483, 0.005212809464829937, 0.005180087468871997, 0.005109367312597701, 0.005008036655737902, 0.0048824062196443085, 0.004737857705157444]
const KS2_beta_patient = 0.9877455919615871
const KS2_dK = [0.014774234539011766, 0.027421363801487738, 0.038183169749769465, 0.04727609533803721, 0.05489357539211571, 0.061208375561011434, 0.06637474630078236, 0.0705303984524335]
const KS2_nlK = [0.014780432326662577, 0.027440542090048974, 0.03821919724766876, 0.047330578768838956, 0.05496665355157508, 0.06129923214320384, 0.06648195952012213, 0.07065217609659635]
const HANK_beta = 0.9810350508981529; const HANK_vphi = 0.7892959971419915
const HANK_dY = [0.002425621631893345, 0.0012111126227469355, 0.0007867698975240949, 0.0004516076427200254, 0.0002719961371235806, 0.00015866808202815428, 9.22807840331099e-05, 5.215734852189905e-05]
const HANK_dC = [0.0024256216318933434, 0.0012111126227469366, 0.000786769897524095, 0.00045160764272002583, 0.00027199613712358065, 0.0001586680820281545, 9.228078403310991e-05, 5.215734852189903e-05]
const HANK_dpi = [0.0019287042002023424, 0.0011047229909627364, 0.0006916025777107582, 0.0004203333760411216, 0.0002621310270469112, 0.00016454095076084567, 0.00010539585450201537, 6.891150390597103e-05]
const HANK_dw = [0.008294773435230022, 0.004165612220963121, 0.002733604124957125, 0.0015950648345713035, 0.0009840868798139302, 0.0005966945374391517, 0.000368271938990591, 0.00022880363748010866]
const HANK_nlY = [0.002536267295883207, 0.0012252991919772683, 0.0007987777267924331, 0.00045437888643996317, 0.0002735628955786334, 0.00015920855103324457, 9.257850580285761e-05, 5.231650100989479e-05]
const HANK_nlC = [0.0024220487031250126, 0.0011885742854151893, 0.0007843338666770894, 0.00044906990025705973, 0.0002714980614948459, 0.00015839606409340234, 9.224526242390674e-05, 5.2174122107562226e-05]
const HANK_nlpi = [0.0019506618999163123, 0.0011063532372662564, 0.0006938384926639006, 0.00042066601334919896, 0.00026234835841336033, 0.0001645681039849919, 0.00010539393580087886, 6.888890726027767e-05]

# functions used by the output-detection tests (they must live in a file)
function _t_nested(; x, y)
    a = max(x, (y + 1) * 2)
    b = min(x, y)
    return (a=a, b=b, c=log(max(a, 1.0)))
end
function _t_multiline(; x, y)
    s = 0.0
    for i in 1:3 s += i end
    return (total = x + s,
            other = y)
end
function _t_branches(; x)
    if x > 0
        return (pos=x, neg=0.0)
    end
    return (pos=0.0, neg=-x)
end

"""
A docstring in front of a block function used to break output detection.
"""
function _t_documented(; x, y)
    z = x * y   # a comment with a return (a=1) in it
    return (z=z,)
end

@testset "SequenceJacobian" begin

@testset "sparse lead/lag operators" begin
    # Q(i,m): ones on diagonal i, first m entries of that diagonal missing
    T = 40
    dense(i, m) = SJ.matrix(SJ.SimpleSparse(Dict((i, m) => 1.0)), T)
    bad = 0
    for i in -3:3, j in -3:3, m in 0:3, n in 0:3
        k, l = SJ.multiply_basis((i, m), (j, n))
        P = dense(i, m) * dense(j, n)
        Q = dense(k, max(l, 0))
        bad += !(P[1:T-8, 1:T-8] ≈ Q[1:T-8, 1:T-8])
    end
    @test bad == 0
    # lead times lag is the identity (this is the case the old rule got wrong)
    @test SJ.multiply_basis((1, 0), (-1, 0)) == (0, 0)
    @test SJ.multiply_basis((-1, 0), (1, 0)) == (0, 1)
end

@testset "output detection" begin
    @test collect(output_list(_t_nested)) == ["a", "b", "c"]
    @test collect(output_list(_t_multiline)) == ["total", "other"]
    @test collect(output_list(_t_branches)) == ["pos", "neg"]
    @test collect(output_list(_t_documented)) == ["z"]
    @test collect(simple(_t_nested).outputs) == ["a", "b", "c"]
end

@testset "small utilities" begin
    g = nonlinspace(10.0, 6, 1.5)
    @test g[1] == 0.0 && g[end] ≈ 10.0 && all(isfinite, g) && issorted(g)
    D = rand(4, 3, 5); X = rand(4, 3, 5)
    @test fast_aggregate(D, X) ≈ [sum(D[t, :, :] .* X[t, :, :]) for t in 1:4]
    a = ImpulseDict(Dict("x" => [1.0, 2.0], "y" => [3.0, 4.0]))
    @test a["x"] == [1.0, 2.0]
    @test collect(keys(a[["x"]])) == ["x"]
    @test (2 * a - a)["y"] == [3.0, 4.0]
    @test SJ.pack(a; force_order=["y", "x"]) == [3.0, 4.0, 1.0, 2.0]
    @test weighted_quantile([3.0, 1.0, 2.0], [0.2, 0.5, 0.3], 0.5) == 1.0
    yq = interpolate_y([1.0, 2.0, 3.0], [1.5, 2.5], [1.0, 4.0, 9.0])
    @test yq ≈ [2.5, 6.5]
end

@testset "grids, Markov chains and moments" begin
    a = agrid(100.0, 5)
    @test a[1] == 0.0 && a[end] ≈ 100.0 && issorted(a)
    y, pi, Pi = markov_rouwenhorst(0.966, 0.5, 3)
    @test isapprox(sum(pi .* y), 1.0; atol=1e-8)
    @test all(isapprox.(sum(Pi, dims=2), 1.0; atol=1e-12))
    @test isapprox(disc_mean(y, pi), 1.0; atol=1e-8)
    @test isapprox(stationary(Pi)' * Pi, stationary(Pi)'; atol=1e-10)
    yt, pit, Pit = markov_tauchen(0.9, 0.2, 5)
    @test isapprox(sum(pit), 1.0; atol=1e-12) && length(yt) == 5
    b = Bijection(Dict("a" => "A"))
    @test b["a"] == "A" && b["x"] == "x"
    x = rand(3, 4, 6); sort!(x, dims=3); x[:, :, 1] .= -1.0
    setmin(x, 0.0)
    @test all(x .>= 0.0)                       # setmin works on arrays of any dimension
end

@testset "bounded steady-state solver" begin
    unknowns = Dict("a" => (0.0, 1.0, 5.0), "b" => 0.5)      # a has bounds, b does not
    names = collect(keys(unknowns))                            # the solver's own ordering
    residual(x) = (d = Dict(zip(names, x)); [d["a"]^2 - 4.0, d["b"] - 1.0])
    sol = SJ.solve_for_unknowns(residual, unknowns, "broyden_custom", Dict(); verbose=false)
    @test isapprox(sol["a"], 2.0; atol=1e-8) && isapprox(sol["b"], 1.0; atol=1e-8)
end

@testset "RBC" begin
    model, ss, unknowns, targets, exogenous = rbc_dag()
    @test isapprox(ss["K"], RBC_K; atol=1e-8) && isapprox(ss["Z"], RBC_Z; atol=1e-8)
    T = 300
    dZ = collect(0.01 .* 0.9 .^ (0:T-1))
    G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
    @test close_to(G["Y"]["Z"] * dZ, RBC_dY)
    @test close_to(G["K"]["Z"] * dZ, RBC_dK)
    @test close_to(G["C"]["Z"] * dZ, RBC_dC)
    lin = solve_impulse_linear(model, ss, unknowns, targets, Dict("Z" => dZ))
    @test close_to(lin["Y"], RBC_dY)
    @test lin["K"] ≈ G["K"]["Z"] * dZ
    nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => dZ); verbose=false)
    @test close_to(nl["Y"], RBC_nlY) && close_to(nl["K"], RBC_nlK)
    big = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => 10 .* dZ); verbose=false, warm_start=true)
    @test close_to(big["Y"], RBC_nl10Y; tol=1e-6)
    @test steady_path_error(model, ss, unknowns, targets; T=10).max_error < 1e-10

    # the same model with labour solved inside a SolvedBlock must give the same answers
    inner  = create_model(collect(model.blocks); name="rbc_inner")
    nested = create_model([solved(inner, Dict("L" => (0.5, 1.5)), ["euler"]; name="rbc_labour_solved", solver="brentq")];
                          name="RBC nested")
    @test isapprox(steady_state(nested, ss)["L"], ss["L"]; atol=1e-8)
    Gn = solve_jacobian(nested, ss, ["K"], ["goods_mkt"], ["Z"]; T=T)
    @test close_to(Gn["Y"]["Z"] * dZ, RBC_dY)
    @test close_to(solve_impulse_linear(nested, ss, ["K"], ["goods_mkt"], Dict("Z" => dZ))["Y"], RBC_dY)
    nln = solve_impulse_nonlinear(nested, ss, ["K"], ["goods_mkt"], Dict("Z" => dZ); verbose=false)
    @test close_to(nln["Y"], RBC_nlY; tol=1e-6) && close_to(nln["K"], RBC_nlK; tol=1e-6)
end

@testset "Krusell-Smith" begin
    ks_ss, ss, model, unknowns, targets, exogenous = ks_dag()
    @test isapprox(ss["beta"], KS_beta; atol=1e-8)
    @test abs(ss["asset_mkt"]) < 1e-8
    T = 300
    dZ = collect(0.01 .* 0.9 .^ (0:T-1))
    hh = het_blocks(model)[1]
    J = jacobian(hh, ss, ["r", "w"], ["A", "C"]; T=T)
    @test isapprox(J["A"]["r"][1, 1], KS_J_A_r_11; atol=1e-6)
    @test isapprox(J["A"]["r"][6, 4], KS_J_A_r_64; atol=1e-6)
    @test isapprox(J["C"]["w"][3, 8], KS_J_C_w_38; atol=1e-6)
    G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
    @test close_to(G["K"]["Z"] * dZ, KS_dK) && close_to(G["C"]["Z"] * dZ, KS_dC)
    # household block alone, nonlinear, with the distribution path kept
    dr = collect(0.001 .* 0.8 .^ (0:T-1))
    pe = impulse_nonlinear(hh, ss, Dict("r" => dr); internals=Dict(hh.name => ["D", "Dbeg_terminal"]))
    @test close_to(pe["A"], KS_peA) && close_to(pe["C"], KS_peC)
    @test size(pe.internals[hh.name]["D"], 1) == T
    @test terminal_distribution_distance(pe, hh.name) < 1e-4
    @test close_to(impulse_linear(hh, ss, Dict("r" => dr))["A"], KS_pelinA)
    # general equilibrium
    lin = solve_impulse_linear(model, ss, unknowns, targets, Dict("Z" => dZ))
    @test close_to(lin["C"], KS_dC)
    nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => dZ); verbose=false)
    @test close_to(nl["K"], KS_nlK) && close_to(nl["C"], KS_nlC)
    # options reach the household block through the model
    loose = steady_state(model, ss; backward_tol=1e-3, forward_tol=1e-3)
    @test abs(loose["A"] - ss["A"]) > 1e-9
    @test fake_news_error(hh, ss, ["r", "w"], ["A", "C"]; T=6).max_error < 1e-4
    # production step size: two-sided differences are closer to brute force than one-sided
    one = fake_news_error(hh, ss, ["r"], ["A", "C"]; T=6, h=1e-4, h_direct=1e-6).max_error
    two = fake_news_error(hh, ss, ["r"], ["A", "C"]; T=6, h=1e-4, h_direct=1e-6, twosided=true).max_error
    @test two < one
end

@testset "Krusell-Smith with two remapped household types" begin
    ks_ss, ss, model, unknowns, targets, exogenous = ks_remapped_dag()
    @test isapprox(ss["beta_patient"], KS2_beta_patient; atol=1e-8)
    T = 300
    dZ = collect(0.01 .* 0.9 .^ (0:T-1))
    G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
    @test close_to(G["K"]["Z"] * dZ, KS2_dK)
    nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => dZ); verbose=false)
    @test close_to(nl["K"], KS2_nlK)
    @test length(het_blocks(model)) == 2
end

@testset "one-asset HANK" begin
    hank_ss, ss, model, unknowns, targets, exogenous = hank_dag()
    @test isapprox(ss["beta"], HANK_beta; atol=1e-7) && isapprox(ss["vphi"], HANK_vphi; atol=1e-7)
    T = 300
    dr = collect(-0.0025 .* 0.6 .^ (0:T-1))
    G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
    @test close_to(G["Y"]["rstar"] * dr, HANK_dY) && close_to(G["C"]["rstar"] * dr, HANK_dC)
    @test close_to(G["pi"]["rstar"] * dr, HANK_dpi) && close_to(G["w"]["rstar"] * dr, HANK_dw)
    lin = solve_impulse_linear(model, ss, unknowns, targets, Dict("rstar" => dr))
    @test close_to(lin["Y"], HANK_dY)
    hh = het_blocks(model)[1]
    nl = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("rstar" => dr); verbose=false, maxit=100,
                                 internals=Dict(hh.name => ["D", "Dbeg_terminal"]))
    @test close_to(nl["Y"], HANK_nlY) && close_to(nl["C"], HANK_nlC) && close_to(nl["pi"], HANK_nlpi)
    @test terminal_distribution_distance(nl, hh.name) < 1e-5
    ch = channel_decomposition(model, ss, G, Dict("rstar" => dr); output="C", direct=["r"])
    @test ch.closure_error < 1e-10
    @test ch.total ≈ G["C"]["rstar"] * dr
    ssd = SJ.extract_ss_dict(hh, ss)
    m = mpc_statistics(hh, ss; transfer="T", amount=0.01, consumption="c", horizons=(1, 4),
                       groups=Dict("constrained" => ssd["a"] .<= ssd["a_grid"][1] + 1e-12))
    @test m.replay_error < 1e-6
    @test 0 < m.stats[1].mean < m.stats[4].mean < 1
    @test m.stats[1].groups["constrained"] > m.stats[1].mean
end


@testset "labor choice (StageBlock)" begin
    model_ss, ss, model, unknowns, targets, exogenous = labor_choice_dag()
    hh = only(b for b in model.blocks if b isa StageBlock)
    @test abs(ss["asset_mkt"]) < 1e-8
    @test abs(ss["goods_mkt"]) < 1e-7                              # Walras' law
    @test 0 < ss["EMP"] < 1
    @test isapprox(sum(ss.internals[hh.name]["prod"]["D"]), 1.0; atol=1e-10)

    # fake-news Jacobian of the stage block against a brute-force Jacobian
    fn = fake_news_error(hh, ss, ["r", "w"], ["A", "C", "NE"]; T=5)
    scale = maximum(maximum(abs.(fn.direct.nesteddict[o][i])) for o in ["A", "C", "NE"], i in ["r", "w"])
    @test fn.max_error / scale < 1e-3

    T = 300
    dZ = collect(0.01 .* ss["Z"] .* 0.9 .^ (0:T-1))
    G = solve_jacobian(model, ss, unknowns, targets, exogenous; T=T)
    lin = solve_impulse_linear(model, ss, unknowns, targets, Dict("Z" => dZ))
    @test lin["Y"] ≈ G["Y"]["Z"] * dZ
    small = solve_impulse_nonlinear(model, ss, unknowns, targets, Dict("Z" => 0.1 .* dZ); verbose=false)
    @test maximum(abs.(10 .* small["Y"][1:50] .- lin["Y"][1:50])) < 2e-2 * maximum(abs.(lin["Y"]))
    @test maximum(abs.(small["goods_mkt"])) < 1e-6                 # Walras' law along the path
    @test steady_path_error(model, ss, unknowns, targets; T=10).max_error < 1e-6
end

@testset "estimation" begin
    # AR(1) observed without measurement error: covariances and likelihood
    T, rho, sigma = 200, 0.8, 0.5
    M = reshape(collect(rho .^ (0:T-1)), T, 1, 1)
    Sigma = all_covariances(M, [sigma])
    @test isapprox(Sigma[1, 1, 1], sigma^2 / (1 - rho^2); rtol=1e-6)
    @test isapprox(Sigma[3, 1, 1], rho^2 * sigma^2 / (1 - rho^2); rtol=1e-6)
    Y = reshape(collect(0.1 .* (1:20) .- 1.0), 20, 1)
    L = log_likelihood(Y, Sigma)
    V = build_full_covariance_matrix(Sigma, [0.0], 20)
    @test isapprox(L, -(logdet(V) + Y[:]' * (V \ Y[:])) / 2; rtol=1e-8)
    @test size(build_MA_representation(Dict("y" => Dict("e" => Matrix(1.0I, T, T))), ["y"], ["e"], T)) == (T, 1, 1)
end

@testset "drawdag" begin
    model, ss, unknowns, targets, exogenous = rbc_dag()
    dot = drawdag(model; exogenous=exogenous, unknowns=unknowns, targets=targets)
    @test startswith(dot, "digraph") && occursin("targets", dot) && endswith(dot, "}")
end

end
println("All tests finished.")
