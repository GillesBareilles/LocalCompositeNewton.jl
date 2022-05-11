function expe_maxquad(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = Float64
    pb = MaxQuadBGLS(Tf)
    x = zeros(Tf, 10) .+ 1.0

    x = [
        0.1291708855755244,
        0.04045443436441793,
        -0.014620528767775722,
        -0.02339789227333125,
        -0.05965860244841998,
        0.2688628230766436,
        -0.07981011132211821,
        -0.1383823364647686,
        -0.08009288114659685,
        -0.039024957152197254,
    ]

    optparams_precomp = OptimizerParams(;
        iterations_limit=2, trace_length=0, time_limit=0.5
    )

    ## Define solvers to be run on problem
    optimdata = OrderedDict()
    time_limit = 0.005

    # Gradient sampling
    o = GradientSampling(x)
    optparams = OptimizerParams(; iterations_limit=100, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_gs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # nsBFGS
    o = NSBFGS{Tf}(; ϵ_opt=1e-15)
    optparams = OptimizerParams(; iterations_limit=300, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_nsbfgs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # Local Newton method
    getx(o, os) = deepcopy(os.x)
    getγ(o, os) = deepcopy(os.γ)
    optimstate_extensions = OrderedDict{Symbol,Function}(:x => getx, :γ => getγ)

    # find the smaller γ which gives maximal structure
    gx = sort(g(pb, x))
    γ = 0.0
    for i in 1:(length(gx) - 1)
        γ += (gx[end - i + 1] - gx[end - i]) * i
    end
    # @show guessstruct_prox(pb, x, γ)
    # @show guessstruct_prox(pb, x, γ-0.01)
    # @show guessstruct_prox(pb, x, γ+0.01)
    # @show guessstruct_prox(pb, x, γ+0.01)
    # @show guessstruct_prox(pb, x, γ/10)
    # @show γ

    o = LocalCompositeNewtonOpt{Tf}(0, 0.0)
    optparams = OptimizerParams(; iterations_limit=5, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    state = initial_state(o, x, pb; γ)
    xfinal_localNewton, tr = NSS.optimize!(
        pb, o, x; state, optparams, optimstate_extensions
    )
    optimdata[o] = tr

    ## Build figures
    xopt = xfinal_localNewton
    Mopt = MaxQuadManifold(pb, [2, 3, 4, 5])
    Fopt = prevfloat(F(pb, xopt))

    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "maxquadBGLS"; NUMEXPS_OUTDIR)
    return true
end
