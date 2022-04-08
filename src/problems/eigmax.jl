function expe_eigmax(NUMEXPS_OUTDIR = NUMEXPS_OUTDIR_DEFAULT)
    n = 20
    m = 25
    pb = get_eigmax_affine(; m, n, seed = 1864)
    x = zeros(n)
    Tf = Float64

    xopt = [-0.29065354221788825, -0.5913012175150199, 0.024727781899874696, -0.03725900302413318, -0.05387465582448637, -0.29513727892621694, 0.3456777615557909, 0.291710323463778, 0.12636034093320853, -0.2743512897606581, -0.7353088734430314, 0.5811490138599301, -0.23867557186411822, 0.4768462996279805, -0.4792234536201222, 0.1324053771135764, -0.010353933273842373, -0.8955538305001317, 0.032694439388782, 0.16534043275844432]
    # Fopt = F(pb, xopt) - 100 * eps(Tf)
    Fopt = prevfloat(2.3299509821980916e+01)
    Mopt = EigmaxManifold(pb, 4)
    # Random.seed!(1643)
    # x = xopt + randn(n) .* 1e-3
    # @show x
    x = [-0.2903621117378083, -0.5906936043347821, 0.022580009190398934, -0.036962726426832536, -0.05311102380035424, -0.29609094669375197, 0.34511861687804846, 0.2917244946010309, 0.126235568114234, -0.2749392842386973, -0.7340388824944295, 0.5802709786578186, -0.23976054387022805, 0.4780418928792374, -0.4803760994489224, 0.1340455884010646, -0.009960099075968799, -0.8955982952380783, 0.03225246196467059, 0.16501043656163675]


    optparams_precomp = OptimizerParams(iterations_limit=2, trace_length=0, time_limit = 0.5)
    time_limit = 1

    ## Define solvers to be run on problem
    optimdata = OrderedDict()

    # Gradient sampling
    o = GradientSampling(m=5, β=1e-4)
    optparams = OptimizerParams(iterations_limit=100, trace_length=50, time_limit=time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_gs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # nsBFGS
    o = NSBFGS{Tf}()
    optparams = OptimizerParams(iterations_limit=300, trace_length=50, time_limit=time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_nsbfgs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # Local Newton method
    getx(o, os) = deepcopy(os.x)
    getγ(o, os) = deepcopy(os.γ)
    optimstate_extensions = OrderedDict{Symbol, Function}(:x => getx, :γ => getγ)

    o = LocalCompositeNewtonOpt{Tf}(0, 0.)
    optparams = OptimizerParams(iterations_limit=5, trace_length=50, time_limit = time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_localNewton, tr = NSS.optimize!(pb, o, x; optparams, optimstate_extensions)
    optimdata[o] = tr


    ## Build figures
    println("Building figures for Eigmax...")

    stepinfo = treatproxsteps(pb, tr, Mopt, xopt)

    optimdatagamma = OrderedDict(
        L"\gamma low(x_k)" => [(itstepinfo.γlow, itstepinfo.distopt) for itstepinfo in stepinfo],
        L"\bar{\gamma}(x_k)" => [(itstepinfo.γup, itstepinfo.distopt) for itstepinfo in stepinfo],
        L"\gamma_k" => [(itstepinfo.γₖ, itstepinfo.distopt) for itstepinfo in stepinfo],
    )
    display(optimdatagamma)
    getabsc_distopt(o, trace) = [ o[2] for o in trace]
    getord_gamma(o, trace) = [ o[1] for o in trace]
    fig = plot_curves(optimdatagamma, getabsc_distopt, getord_gamma;
                      xlabel = L"\| x_k - x^\star\|",
                      ylabel = L"",
                      # nmarks = 1000,
                      xmode = "log"
                      )
    savefig(fig, joinpath(NUMEXPS_OUTDIR, "eigmax_gamma"))

    # Suboptimality
    getabsc_time(optimizer, trace) = [state.time for state in trace]
    getord_subopt(optimizer, trace) = [state.Fx-Fopt for state in trace]
    fig = plot_curves(optimdata, getabsc_time, getord_subopt;
                      xlabel = "time (s)",
                      ylabel = L"F(x_k) - F^\star",
                      nmarks = 1000,
                      )
    savefig(fig, joinpath(NUMEXPS_OUTDIR, "eigmax_time_subopt"))

    # getabsc_it(optimizer, trace) = [state.it for state in trace]
    # fig = plot_curves(optimdata, getabsc_it, getord_subopt;
    #                   xlabel = "iteration",
    #                   ylabel = L"F(x_k) - F^\star",
    #                   nmarks = 1000,
    #                   )
    # savefig(fig, joinpath(NUMEXPS_OUTDIR, "eigmax_it_subopt"))
    return
end

function treatproxsteps(pb, tr, Mopt::EigmaxManifold, xopt)
    stepinfo = Any[]
    for os in tr[1:end]
        x = os.additionalinfo.x
        gx = eigvals(NSP.g(pb, x))

        # Computing steps low, up
        r = Mopt.eigmult.r
        γlow, γup = get_γlowγupmax(gx, r)
        γₖ = os.additionalinfo.γ
        distopt = norm(x - xopt)
        push!(stepinfo, (;γlow, γup, γₖ, distopt))
    end
    return stepinfo
end
