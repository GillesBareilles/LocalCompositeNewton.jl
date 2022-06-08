function expe_maxquad(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = Float64
    pb = MaxQuadBGLS(Tf)

    x = [
        0.12625658077472543,
        0.03437830256204086,
        0.006857198326981548,
        -0.026360658246337897,
        -0.06729492268974148,
        0.2783995007519937,
        -0.07421866454469361,
        -0.1385240478372969,
        -0.08403122312533247,
        -0.03858030977273088,
    ]
    x .+= 1e-2 .* ones(Tf, 10)

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
    optparams = OptimizerParams(; iterations_limit=700, trace_length=50, time_limit)
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

    # @show xopt
    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "maxquadBGLS"; NUMEXPS_OUTDIR)
    return true
end
