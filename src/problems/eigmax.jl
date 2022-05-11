function expe_eigmax(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    n = 25
    m = 50
    pb = get_eigmax_affine(; m, n, seed=1864)
    x = zeros(n)
    Tf = Float64

    x = [0.18843321959369272, 0.31778063128576134, 0.34340066698932187, -0.27805652811628134, -0.1340243453861452, -0.12921798176305369, -0.5566692206939368, -0.6007421833719635, 0.05910386724008742, 0.17705864693916648, 0.08556420932871216, -0.026666254662448905, -0.23677377353260096, -0.48199437746045676, 0.06585075102257752, 0.04851608933735588, -0.3925094708809553, -0.24927524067693352, 0.5381266955502098, 0.2599737695610786, -0.5646166025020284, 0.1550051571713463, -0.2641217487440864, 0.3668468331373211, -0.2080390109713874]
    x += 1e-3 * ones(n)

    optparams_precomp = OptimizerParams(;
        iterations_limit=2, trace_length=0, time_limit=0.5
    )
    time_limit = 1

    ## Define solvers to be run on problem
    optimdata = OrderedDict()

    # Gradient sampling
    o = GradientSampling(x)
    optparams = OptimizerParams(;
        iterations_limit=100, trace_length=50, time_limit=time_limit
    )
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_gs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # nsBFGS
    o = NSBFGS{Tf}()
    optparams = OptimizerParams(;
        iterations_limit=300, trace_length=50, time_limit=time_limit
    )
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_nsbfgs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    @show xfinal_nsbfgs
    @show eigvals(g(pb, xfinal_nsbfgs))[end-10:end]
    # Local Newton method
    getx(o, os) = deepcopy(os.x)
    getγ(o, os) = deepcopy(os.γ)
    optimstate_extensions = OrderedDict{Symbol,Function}(:x => getx, :γ => getγ)

    # find the smaller γ which gives maximal structure
    # Note that the maximum structure here is r=6 and not b=25.
    # Indeed, the codimension of the structure manifold exceeds n=25 for r>6.
    gx = eigvals(g(pb, x))
    γ = 0.0
    for i in 1:5
        @show (gx[end-i+1] - gx[end-i]) * i
        γ += (gx[end-i+1] - gx[end-i]) * i
    end
    @show size(g(pb, x))
    @show guessstruct_prox(pb, x, γ)
    @show guessstruct_prox(pb, x, γ-0.01)
    @show guessstruct_prox(pb, x, γ+0.01)
    @show guessstruct_prox(pb, x, γ+0.01)
    @show guessstruct_prox(pb, x, γ/10)
    @show γ

    o = LocalCompositeNewtonOpt{Tf}(0, 0.0)
    optparams = OptimizerParams(;
        iterations_limit=10, trace_length=50, time_limit=time_limit
    )
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    state = initial_state(o, x, pb; γ)
    xfinal_localNewton, tr = NSS.optimize!(pb, o, x; state, optparams, optimstate_extensions)
    optimdata[o] = tr

    ## Build figures
    xopt = xfinal_localNewton
    Mopt = EigmaxManifold(pb, 3)
    Fopt = prevfloat(F(pb, xopt))

    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "eigmax"; NUMEXPS_OUTDIR)

    return true
end
