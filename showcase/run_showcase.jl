# run_showcase.jl: results for the repository page.
#
#   julia showcase/run_showcase.jl                 (from the repository root)
#
# Runs the test models on finer grids and writes
#   showcase/RESULTS.md        tables and figures, ready to link from the README
#   showcase/figures/*.png     figures
#   showcase/numbers.txt       every number printed below
#
# The first run sets up a small environment in showcase/ with this package and Plots.
# Set RUN_TWO_ASSET=0 to skip the two-asset model (its steady state takes minutes).

using Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(path=joinpath(@__DIR__, ".."))
    Pkg.add("Plots")
end
Pkg.instantiate()

using SequenceJacobian, Plots, Printf, Dates
gr()
default(linewidth=2, framestyle=:box, grid=true, legend=:best, titlefontsize=11,
        guidefontsize=9, tickfontsize=8, legendfontsize=8, dpi=150)

const OUT = @__DIR__
const FIG = joinpath(OUT, "figures")
mkpath(FIG)
const RUN_TWO_ASSET = get(ENV, "RUN_TWO_ASSET", "1") != "0"
const T = 300
const H = 40                                   # quarters shown in figures

log_io = open(joinpath(OUT, "numbers.txt"), "w")
function say(args...)
    s = string(args...)
    println(s); println(log_io, s); flush(log_io)
end

pct(x, s) = 100 .* x ./ s                       # percent of steady state
pp(x)     = 100 .* x                            # percentage points
q(t)      = 0:t-1

function panel(title, ylabel, curves; h=H)
    p = plot(title=title, xlabel="quarters", ylabel=ylabel)
    for (label, y, style) in curves
        plot!(p, q(h), y[1:h]; label=label, linestyle=style)
    end
    hline!(p, [0.0]; color=:black, linewidth=0.5, label="")
    return p
end

relerr(fn) = fn.max_error / maximum(maximum(abs.(m)) for d in values(fn.direct.nesteddict) for m in values(d))
target_residual(ss, targets) = maximum(abs(ss[t]) for t in targets)

function wealth_stats(D, a_grid)
    w = vec(sum(D, dims=1)); a = collect(a_grid)
    W = cumsum(w .* a) ./ sum(w .* a); P = cumsum(w)
    gini = 1 - sum((W .+ [0; W[1:end-1]]) .* w)
    top10 = 1 - W[findfirst(>=(0.9), P)]
    return (gini=gini, top10=top10, at_zero=w[1], P=P, W=W)
end

# ---------------------------------------------------------------------------------
say("SequenceJacobian.jl showcase  (", SSJ_VERSION, ", Julia ", VERSION, ", ", Dates.today(), ")")
say("CPU: ", strip(Sys.cpu_info()[1].model), ", Julia threads: ", Threads.nthreads())

# ---- warm-up: compile on the small test models, so timings below exclude compilation
say("\nWarm-up on the small test models (compilation)...")
let
    m = rbc_dag();          solve_jacobian(m[1], m[2], m[3], m[4], m[5]; T=T)
    for f in (ks_dag, hank_dag, labor_choice_dag)
        mss, ss, model, u, tg, ex = f()
        G = solve_jacobian(model, ss, u, tg, ex; T=T)
        solve_impulse_nonlinear(model, ss, u, tg, Dict(ex[end] => 1e-4 .* 0.9 .^ q(T)); verbose=false)
    end
end

timing = Any[]          # (model, grid, steady state, GE Jacobian, nonlinear)
accuracy = Any[]        # (model, ss residual, steady path, fake news, terminal distance)

# =================================================================================
# 1. RBC
# =================================================================================
say("\n== RBC ==")
t_ss = @elapsed (rbc, ss_rbc, u, tg, ex) = rbc_dag()
t_G  = @elapsed G = solve_jacobian(rbc, ss_rbc, u, tg, ex; T=T)
dZ   = 0.01 .* ss_rbc["Z"] .* 0.9 .^ q(T)
t_nl = @elapsed nl = solve_impulse_nonlinear(rbc, ss_rbc, u, tg, Dict("Z" => dZ); verbose=false)
push!(timing, ("RBC", "—", t_ss, t_G, t_nl))
push!(accuracy, ("RBC", target_residual(ss_rbc, tg), steady_path_error(rbc, ss_rbc, u, tg; T=20).max_error, NaN, NaN))
p = plot([panel("Output", "% of s.s.", [("linear", pct(G["Y"]["Z"]*dZ, ss_rbc["Y"]), :solid), ("nonlinear", pct(nl["Y"], ss_rbc["Y"]), :dash)]),
          panel("Consumption", "% of s.s.", [("linear", pct(G["C"]["Z"]*dZ, ss_rbc["C"]), :solid), ("nonlinear", pct(nl["C"], ss_rbc["C"]), :dash)]),
          panel("Capital", "% of s.s.", [("linear", pct(G["K"]["Z"]*dZ, ss_rbc["K"]), :solid), ("nonlinear", pct(nl["K"], ss_rbc["K"]), :dash)]),
          panel("Hours", "% of s.s.", [("linear", pct(G["L"]["Z"]*dZ, ss_rbc["L"]), :solid), ("nonlinear", pct(nl["L"], ss_rbc["L"]), :dash)])]...,
         layout=(2, 2), size=(900, 600), plot_title="RBC: 1% TFP shock, persistence 0.9")
savefig(p, joinpath(FIG, "rbc_tfp.png"))
say(@sprintf("Impact response of Y: linear %.4f%%, nonlinear %.4f%%", pct(G["Y"]["Z"]*dZ, ss_rbc["Y"])[1], pct(nl["Y"], ss_rbc["Y"])[1]))

# =================================================================================
# 2. Krusell-Smith (7 income states, 500 asset points, as in the Python notebook)
# =================================================================================
say("\n== Krusell-Smith (nS = 7, nA = 500) ==")
t_ss = @elapsed (ks_ss_m, ss_ks, ks, u, tg, ex) = ks_dag(nS=7, nA=500)
hh_ks = het_blocks(ks)[1]
t_G  = @elapsed G = solve_jacobian(ks, ss_ks, u, tg, ex; T=T)
dZ   = 0.01 .* ss_ks["Z"] .* 0.9 .^ q(T)
t_nl = @elapsed nl = solve_impulse_nonlinear(ks, ss_ks, u, tg, Dict("Z" => dZ); verbose=false,
                                             internals=Dict(hh_ks.name => ["D", "Dbeg_terminal"]))
fn = fake_news_error(hh_ks, ss_ks, ["r", "w"], ["A", "C"]; T=10)
push!(timing, ("Krusell-Smith", "7 × 500", t_ss, t_G, t_nl))
push!(accuracy, ("Krusell-Smith", abs(ss_ks["asset_mkt"]), steady_path_error(ks, ss_ks, u, tg; T=20).max_error,
                 relerr(fn), terminal_distribution_distance(nl, hh_ks.name)))
say(@sprintf("beta = %.6f, K/Y = %.3f", ss_ks["beta"], ss_ks["K"] / ss_ks["Y"]))

p = plot([panel("Output", "% of s.s.", [("linear", pct(G["Y"]["Z"]*dZ, ss_ks["Y"]), :solid), ("nonlinear", pct(nl["Y"], ss_ks["Y"]), :dash)]),
          panel("Consumption", "% of s.s.", [("linear", pct(G["C"]["Z"]*dZ, ss_ks["C"]), :solid), ("nonlinear", pct(nl["C"], ss_ks["C"]), :dash)]),
          panel("Capital", "% of s.s.", [("linear", pct(G["K"]["Z"]*dZ, ss_ks["K"]), :solid), ("nonlinear", pct(nl["K"], ss_ks["K"]), :dash)]),
          panel("Interest rate", "pp", [("linear", pp(G["r"]["Z"]*dZ), :solid), ("nonlinear", pp(nl["r"]), :dash)])]...,
         layout=(2, 2), size=(900, 600), plot_title="Krusell-Smith: 1% TFP shock, persistence 0.9")
savefig(p, joinpath(FIG, "ks_tfp.png"))

# the fake-news Jacobian: columns of dC_t / dr_s
J = jacobian(hh_ks, ss_ks, ["r"], ["C"]; T=T)
JC = make_matrix(J["C"]["r"], T)
p = plot(title="Household Jacobian ∂C_t / ∂r_s (Krusell-Smith)", xlabel="t (quarters)", ylabel="response of C_t", size=(800, 450))
for s in (0, 25, 50, 75, 100)
    plot!(p, 0:149, JC[1:150, s+1]; label="s = $s")
end
hline!(p, [0.0]; color=:black, linewidth=0.5, label="")
savefig(p, joinpath(FIG, "ks_jacobian_columns.png"))

# wealth distribution
ssd = SequenceJacobian.extract_ss_dict(hh_ks, ss_ks)
ws = wealth_stats(ssd["D"], ssd["a_grid"])
say(@sprintf("Wealth: Gini %.3f, top 10%% share %.3f, share at the borrowing limit %.3f", ws.gini, ws.top10, ws.at_zero))
p = plot([0; ws.P], [0; ws.W]; label="Lorenz curve", xlabel="population share", ylabel="wealth share",
         title="Steady-state wealth distribution (Krusell-Smith)", size=(550, 500), aspect_ratio=:equal, xlims=(0, 1), ylims=(0, 1))
plot!(p, [0, 1], [0, 1]; color=:black, linewidth=0.5, label="equality")
savefig(p, joinpath(FIG, "ks_lorenz.png"))

# =================================================================================
# 3. One-asset HANK (7 income states, 500 asset points, as in the Python notebook)
# =================================================================================
say("\n== One-asset HANK (nS = 7, nA = 500) ==")
t_ss = @elapsed (hank_ss_m, ss_h, hank, u, tg, ex) = hank_dag(nS=7, nA=500)
hh_h = het_blocks(hank)[1]
t_G  = @elapsed G = solve_jacobian(hank, ss_h, u, tg, ex; T=T)
dr   = -0.0025 .* 0.6 .^ q(T)                                     # 25 bp cut in rstar
t_nl = @elapsed nl = solve_impulse_nonlinear(hank, ss_h, u, tg, Dict("rstar" => dr); verbose=false,
                                             internals=Dict(hh_h.name => ["D", "Dbeg_terminal"]))
fn = fake_news_error(hh_h, ss_h, ["r", "w", "Div", "Tax"], ["A", "C", "NE"]; T=10)
push!(timing, ("One-asset HANK", "7 × 500", t_ss, t_G, t_nl))
push!(accuracy, ("One-asset HANK", target_residual(ss_h, tg), steady_path_error(hank, ss_h, u, tg; T=20).max_error,
                 relerr(fn), terminal_distribution_distance(nl, hh_h.name)))
say(@sprintf("beta = %.6f, vphi = %.4f", ss_h["beta"], ss_h["vphi"]))

p = plot([panel("Output", "% of s.s.", [("linear", pct(G["Y"]["rstar"]*dr, ss_h["Y"]), :solid), ("nonlinear", pct(nl["Y"], ss_h["Y"]), :dash)]),
          panel("Consumption", "% of s.s.", [("linear", pct(G["C"]["rstar"]*dr, ss_h["C"]), :solid), ("nonlinear", pct(nl["C"], ss_h["C"]), :dash)]),
          panel("Inflation", "pp (quarterly)", [("linear", pp(G["pi"]["rstar"]*dr), :solid), ("nonlinear", pp(nl["pi"]), :dash)]),
          panel("Real rate", "pp (quarterly)", [("linear", pp(G["r"]["rstar"]*dr), :solid), ("nonlinear", pp(nl["r"]), :dash)])]...,
         layout=(2, 2), size=(900, 600), plot_title="One-asset HANK: 25 bp cut in the policy rate, persistence 0.6")
savefig(p, joinpath(FIG, "hank_monetary.png"))

# nonlinearity: 100 bp cut and 100 bp hike against the linear response
dr100 = 4 .* dr
cut  = solve_impulse_nonlinear(hank, ss_h, u, tg, Dict("rstar" => dr100);  verbose=false, warm_start=true, maxit=100)
hike = solve_impulse_nonlinear(hank, ss_h, u, tg, Dict("rstar" => -dr100); verbose=false, warm_start=true, maxit=100)
lin  = pct(G["Y"]["rstar"] * dr100, ss_h["Y"])
p = panel("Output response to a 100 bp change in the policy rate", "% of s.s.",
          [("linear", lin, :solid), ("nonlinear, cut", pct(cut["Y"], ss_h["Y"]), :dash),
           ("nonlinear, hike (sign flipped)", -pct(hike["Y"], ss_h["Y"]), :dot)]; h=20)
plot!(p, size=(800, 450))
savefig(p, joinpath(FIG, "hank_nonlinearity.png"))
say(@sprintf("100 bp: impact on Y linear %.4f%%, cut %.4f%%, hike %.4f%%", lin[1], pct(cut["Y"], ss_h["Y"])[1], pct(hike["Y"], ss_h["Y"])[1]))

# channels of the consumption response
ch = channel_decomposition(hank, ss_h, G, Dict("rstar" => dr); output="C", direct=["r"])
curves = Any[("total", pct(ch.total, ss_h["C"]), :solid), ("direct: real rate", pct(ch.direct, ss_h["C"]), :dash)]
channel_labels = Dict("w" => "wages", "Div" => "dividends", "Tax" => "taxes")
for x in sort(collect(keys(ch.contributions)))
    x == "r" && continue
    push!(curves, ("via " * get(channel_labels, x, x), pct(ch.contributions[x], ss_h["C"]), :dot))
end
p = panel("Consumption response by channel (one-asset HANK, 25 bp cut)", "% of s.s. C", curves; h=20)
plot!(p, size=(800, 450))
savefig(p, joinpath(FIG, "hank_channels.png"))
direct_share = ch.direct[1] / ch.total[1]
say(@sprintf("Consumption on impact: total %.4f%%, direct share %.1f%%, adding-up error %.1e",
             pct(ch.total, ss_h["C"])[1], 100 * direct_share, ch.closure_error))

# MPCs
ssd = SequenceJacobian.extract_ss_dict(hh_h, ss_h)
labor_income = sum(ssd["D"] .* (ss_h["w"] .* reshape(ssd["e_grid"], :, 1) .* ssd["n"]))
constrained  = ssd["a"] .<= ssd["a_grid"][1] + 1e-12
m = mpc_statistics(hh_h, ss_h; transfer="T", amount=0.03 * labor_income, consumption="c",
                   horizons=(1, 4), groups=Dict("constrained" => constrained))
mpc_rows = [(h, m.stats[h].mean, m.stats[h].median, m.stats[h].p90, m.stats[h].high_mpc_share, m.stats[h].groups["constrained"]) for h in (1, 4)]
say(@sprintf("Quarterly MPC: mean %.3f, median %.3f; annual MPC mean %.3f; share constrained %.3f",
             m.stats[1].mean, m.stats[1].median, m.stats[4].mean, m.group_mass["constrained"]))
a_grid = ssd["a_grid"]; nS = size(m.mpc[1], 1)
p = plot(title="Quarterly MPC by wealth and income (one-asset HANK)", xlabel="assets", ylabel="MPC",
         xscale=:log10, size=(800, 450), legend=:topright)
for s in unique([1, cld(nS, 2), nS])
    plot!(p, a_grid[2:end], m.mpc[1][s, 2:end]; label="income state $s of $nS")
end
savefig(p, joinpath(FIG, "hank_mpc.png"))

# =================================================================================
# 4. Labor choice (stage block)
# =================================================================================
say("\n== Labor choice, stage block (nZ = 5, nA = 100) ==")
t_ss = @elapsed (lc_ss_m, ss_lc, lc, u, tg, ex) = labor_choice_dag(nZ=5, nA=100)
hh_lc = only(b for b in lc.blocks if b isa StageBlock)
t_G  = @elapsed G = solve_jacobian(lc, ss_lc, u, tg, ex; T=T)
dZ   = 0.01 .* ss_lc["Z"] .* 0.9 .^ q(T)
t_nl = @elapsed nl = solve_impulse_nonlinear(lc, ss_lc, u, tg, Dict("Z" => dZ); verbose=false)
fn = fake_news_error(hh_lc, ss_lc, ["r", "w"], ["A", "C", "NE"]; T=10)
push!(timing, ("Labor choice (stage block)", "2 × 5 × 100", t_ss, t_G, t_nl))
push!(accuracy, ("Labor choice (stage block)", target_residual(ss_lc, tg), steady_path_error(lc, ss_lc, u, tg; T=20).max_error,
                 relerr(fn), NaN))
say(@sprintf("beta = %.6f, employment rate %.3f", ss_lc["beta"], ss_lc["EMP"]))
p = plot([panel("Output", "% of s.s.", [("linear", pct(G["Y"]["Z"]*dZ, ss_lc["Y"]), :solid), ("nonlinear", pct(nl["Y"], ss_lc["Y"]), :dash)]),
          panel("Employment rate", "pp", [("linear", pp(G["EMP"]["Z"]*dZ), :solid), ("nonlinear", pp(nl["EMP"]), :dash)]),
          panel("Consumption", "% of s.s.", [("linear", pct(G["C"]["Z"]*dZ, ss_lc["C"]), :solid), ("nonlinear", pct(nl["C"], ss_lc["C"]), :dash)]),
          panel("Capital", "% of s.s.", [("linear", pct(G["K"]["Z"]*dZ, ss_lc["K"]), :solid), ("nonlinear", pct(nl["K"], ss_lc["K"]), :dash)])]...,
         layout=(2, 2), size=(900, 600), plot_title="Labor choice (stage block): 1% TFP shock")
savefig(p, joinpath(FIG, "labor_choice_tfp.png"))

# =================================================================================
# 5. Two-asset HANK (grids of the Python example)
# =================================================================================
if RUN_TWO_ASSET
    say("\n== Two-asset HANK (nZ = 3, nB = 50, nA = 70) ==")
    t_ss = @elapsed (ta_ss_m, ss_ta, ta, u, tg, ex) = two_asset_dag()
    t_G  = @elapsed G = solve_jacobian(ta, ss_ta, u, tg, ex; T=T)
    drs  = -0.0025 .* 0.6 .^ q(T)
    push!(timing, ("Two-asset HANK", "3 × 50 × 70", t_ss, t_G, NaN))
    push!(accuracy, ("Two-asset HANK", target_residual(ss_ta, tg), NaN, NaN, NaN))
    p = plot([panel("Output", "% of s.s.", [("linear", pct(G["Y"]["rstar"]*drs, ss_ta["Y"]), :solid)]),
              panel("Consumption", "% of s.s.", [("linear", pct(G["C"]["rstar"]*drs, ss_ta["C"]), :solid)]),
              panel("Investment", "% of s.s.", [("linear", pct(G["I"]["rstar"]*drs, ss_ta["I"]), :solid)]),
              panel("Inflation", "pp (quarterly)", [("linear", pp(G["pi"]["rstar"]*drs), :solid)])]...,
             layout=(2, 2), size=(900, 600), legend=false, plot_title="Two-asset HANK: 25 bp cut in the policy rate")
    savefig(p, joinpath(FIG, "two_asset_monetary.png"))
    say(@sprintf("Peak responses: Y %.4f%%, C %.4f%%, I %.4f%%",
                 maximum(pct(G["Y"]["rstar"]*drs, ss_ta["Y"])), maximum(pct(G["C"]["rstar"]*drs, ss_ta["C"])),
                 maximum(pct(G["I"]["rstar"]*drs, ss_ta["I"]))))
end

# =================================================================================
# Tables and RESULTS.md
# =================================================================================
fmt_t(x) = isnan(x) ? "—" : (x < 1 ? @sprintf("%.2f s", x) : @sprintf("%.1f s", x))
fmt_e(x) = isnan(x) ? "—" : @sprintf("%.1e", x)

say("\nTimings (seconds, after compilation): model, grid, steady state, GE Jacobian, nonlinear")
for r in timing; say(join(string.(r), " | ")); end
say("\nAccuracy: model, s.s. residual, steady path, fake-news rel. error, terminal distance")
for r in accuracy; say(join(string.(r), " | ")); end

open(joinpath(OUT, "RESULTS.md"), "w") do io
    println(io, "# Results\n")
    println(io, "Produced by [`showcase/run_showcase.jl`](run_showcase.jl) with SequenceJacobian.jl $(SSJ_VERSION) on Julia $(VERSION). ",
                "The models are the toolbox's test models (`src/examples/`); Krusell-Smith and one-asset HANK use the finer grids of the ",
                "Python toolbox's notebooks. Horizon T = $T quarters.\n")

    println(io, "## Speed\n")
    println(io, "Seconds after compilation, on ", strip(Sys.cpu_info()[1].model), ". *Steady state* includes building the model; ",
                "*GE Jacobian* is `solve_jacobian` with T = $T; *nonlinear* is `solve_impulse_nonlinear` for the shocks in the figures below.\n")
    println(io, "| Model | Household grid | Steady state | GE Jacobian | Nonlinear transition |")
    println(io, "|---|---|---|---|---|")
    for (mname, grid, a, b, c) in timing
        println(io, "| $mname | $grid | $(fmt_t(a)) | $(fmt_t(b)) | $(fmt_t(c)) |")
    end

    println(io, "\n## Accuracy checks\n")
    println(io, "* *Steady-state residual*: largest equilibrium condition at the steady state.")
    println(io, "* *Steady path*: largest equilibrium condition when the steady state is fed through the nonlinear transition (`steady_path_error`).")
    println(io, "* *Fake news vs brute force*: largest difference between the household Jacobian from the fake-news algorithm and one from ",
                "nonlinear transitions, relative to the largest entry (`fake_news_error`, T = 10).")
    println(io, "* *Terminal distance*: largest deviation of the distribution from its steady state after the last date of the nonlinear transition.\n")
    println(io, "| Model | Steady-state residual | Steady path | Fake news vs brute force | Terminal distance |")
    println(io, "|---|---|---|---|---|")
    for (mname, a, b, c, d) in accuracy
        println(io, "| $mname | $(fmt_e(a)) | $(fmt_e(b)) | $(fmt_e(c)) | $(fmt_e(d)) |")
    end

    println(io, "\n## How to read the figures\n")
    println(io, "All shocks are unexpected at date 0 and fade out geometrically. Responses are deviations from the steady state: ",
                "quantities in percent of their steady-state value, rates in percentage points per quarter. Solid lines are the ",
                "first-order (linear) solution computed from the sequence-space Jacobians; dashed lines are the exact nonlinear ",
                "perfect-foresight solution of the same model. Where the two coincide, the linear solution is accurate for that shock.\n")
    println(io, "## RBC\n")
    println(io, "The representative-agent benchmark: a 1% increase in total factor productivity that fades with persistence 0.9. ",
                "There is no household heterogeneity, so this model solves in milliseconds and is the reference point for the models below.\n")
    println(io, "![RBC](figures/rbc_tfp.png)\n")
    println(io, "## Krusell-Smith\n")
    println(io, "The same productivity shock in an economy where households face uninsurable income risk and a borrowing limit ",
                "(7 income states, 500 asset grid points). Aggregate saving now comes from the whole distribution of households.\n")
    println(io, "![Krusell-Smith](figures/ks_tfp.png)\n")
    println(io, "Columns of the household Jacobian of consumption with respect to the interest rate: an expected change ",
                "in r at date s moves consumption before s (anticipation) and after it.\n")
    println(io, "![Jacobian columns](figures/ks_jacobian_columns.png)\n")
    println(io, "The Lorenz curve plots the share of total wealth held by the poorest x% of households. The further it bows below ",
                "the 45-degree line, the more unequal the distribution.")
    println(io, @sprintf("Steady-state wealth: Gini %.3f, top 10%% share %.3f, share at the borrowing limit %.3f.\n", ws.gini, ws.top10, ws.at_zero))
    println(io, "![Lorenz curve](figures/ks_lorenz.png)\n")

    println(io, "## One-asset HANK\n")
    println(io, "A New Keynesian model with heterogeneous households who choose how much to work and save, sticky prices, ",
                "a Taylor rule and a government that adjusts taxes. The shock is a 25 basis point cut in the policy rate ",
                "that fades with persistence 0.6.\n")
    println(io, "![HANK monetary shock](figures/hank_monetary.png)\n")
    println(io, "Linear and nonlinear responses to a 100 basis point change in the policy rate. The hike is shown with its sign ",
                "flipped, so that in a linear model the three lines would coincide; the gaps measure the nonlinearity.")
    println(io, @sprintf("On impact, output rises by %.2f%% after the cut and falls by %.2f%% after the hike; the linear solution gives %.2f%%.\n",
                         pct(cut["Y"], ss_h["Y"])[1], -pct(hike["Y"], ss_h["Y"])[1], lin[1]))
    println(io, "![Nonlinearity](figures/hank_nonlinearity.png)\n")
    println(io, "Channels of the consumption response (`channel_decomposition`). Households respond to the real interest rate ",
                "directly, and indirectly to the changes in wages, dividends and taxes that the rate cut causes in general ",
                "equilibrium. The lines add up to the total.")
    println(io, @sprintf("On impact, the direct effect of the real rate accounts for %.0f%% of the total.\n", 100 * direct_share))
    println(io, "![Channels](figures/hank_channels.png)\n")
    println(io, "Marginal propensities to consume out of a one-time transfer of 3% of mean quarterly labor income (`mpc_statistics`):\n")
    println(io, "| Horizon | Mean | Median | 90th percentile | Share with MPC ≥ 0.5 | Mean, constrained households |")
    println(io, "|---|---|---|---|---|---|")
    for (h, a, b, c, d, e) in mpc_rows
        println(io, @sprintf("| %d quarter%s | %.3f | %.3f | %.3f | %.3f | %.3f |", h, h == 1 ? "" : "s", a, b, c, d, e))
    end
    println(io, @sprintf("\nShare of households at the borrowing constraint: %.3f.\n", m.group_mass["constrained"]))
    println(io, "The quarterly MPC across the wealth distribution for the lowest, middle and highest income state. Households ",
                "with little wealth spend a large part of a transfer at once; the MPC falls as wealth rises.\n")
    println(io, "![MPCs](figures/hank_mpc.png)\n")

    println(io, "## Labor choice (stage block)\n")
    println(io, "Households receive a productivity shock, choose whether to work (logit, with taste shocks), then consume and save. ",
                "The household is written as a `StageBlock` (`src/examples/LaborChoice.jl`).\n")
    println(io, "![Labor choice](figures/labor_choice_tfp.png)\n")

    if RUN_TWO_ASSET
        println(io, "## Two-asset HANK\n")
        println(io, "Households hold a liquid asset and an illiquid asset that is costly to adjust, so many of them are ",
                    "wealthy but hand-to-mouth. Firms invest with adjustment costs; prices and wages are sticky. Response to a ",
                    "25 basis point cut in the policy rate, first-order solution.\n")
        println(io, "![Two-asset HANK](figures/two_asset_monetary.png)\n")
    end
end

say("\nWrote ", joinpath(OUT, "RESULTS.md"), " and ", length(readdir(FIG)), " figures in ", FIG)
close(log_io)
