function expe_nonconvex_maxquad(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = Float64
    n = 50

    b = zeros(n)
    b[1] = -1
    pb = MaxQuadPb{Tf}(2, 2,
        Vector{Matrix{Tf}}([Diagonal(0.5 .* ones(n)), Diagonal(-0.5 * ones(n))]),
        Vector{Vector{Tf}}([zeros(n), b]),
        Vector{Tf}([0, 2])
    )
    x = zeros(n)
    x[1] = 3

    optparams_precomp = OptimizerParams(;
        iterations_limit=2, trace_length=0, time_limit=0.5
    )

    ## Define solvers to be run on problem
    optimdata = OrderedDict()
    time_limit = 0.004

    # Gradient sampling
    o = GradientSampling(x)
    optparams = OptimizerParams(; iterations_limit=100, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_gs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # nsBFGS
    o = NSBFGS{Tf}(; ϵ_opt=1e-15)
    optparams = OptimizerParams(; iterations_limit=700, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_nsbfgs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # Local Newton method
    getx(o, os, optimstate_additionalinfo) = deepcopy(os.x)
    getγ(o, os, optimstate_additionalinfo) = deepcopy(os.γ)
    optimstate_extensions = OrderedDict{Symbol,Function}(:x => getx, :γ => getγ)

    # find the smaller γ which gives maximal structure
    gx = sort(g(pb, x))
    γ = 0.0
    for i in 1:(length(gx) - 1)
        γ += (gx[end - i + 1] - gx[end - i]) * i
    end
    γ *= 2

    o = LocalCompositeNewtonOpt{Tf}()
    optparams = OptimizerParams(; iterations_limit=5, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    state = initial_state(o, x, pb; γ)
    xfinal_localNewton, tr = NSS.optimize!(
        pb, o, x; state, optparams, optimstate_extensions
    )
    optimdata[o] = tr

    ## Build figures
    xopt = xfinal_localNewton
    Mopt = MaxQuadManifold(pb, [1, 2])
    Fopt = prevfloat(F(pb, xopt))

    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "nonconvex-maxquad"; NUMEXPS_OUTDIR, plotgamma = false)
    return true
end
