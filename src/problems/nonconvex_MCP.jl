### instances
function expe_nonconvex_MCP(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = Float64
    # pb = get_2dMCP()

    # x = zeros(2)
    # x[2] = 1.3

    n = 20
    pb = NSP.get_MCP_instance(; n)
    # x = zeros(n)
    xopt = Tf[1.0565430483631981, 4.6912765109580335e-15, 1.0846386912732007, -5.773151562317679e-15, 2.8697105282238926, -0.9251005174218158, 1.2650147471592115, 2.9000555111381007e-15, 6.581193981293898e-16, 3.810480252553204e-15, 1.8829517570149947, -0.21004328221763952, 0.07366979530380999, 9.567427321537056e-15, -1.9084044083406194e-14, 1.272304858773118, 0.3915763694450909, 2.3047400911538185, 2.424552682252719e-15, -9.018335226297223e-16]
    x = xopt .+ 1e-1


    # x = Tf[1, 0, 1, 0, 3, -1, 1, 0, 0, 0, 2, -0.2, 0.08, 0, 0, 1, 0.4, 2, 0, 0]
    # # nz_coords = BitVector(map(t->t==0, x))
    # nz_coords = Bool[0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


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
    # find the smaller γ which gives maximal structure
    γ = maximum(map(t -> abs(t)^2 / (pb.β * pb.λ^2), x))

    o = LocalCompositeNewtonOpt{Tf}(; decreasefactor = 9.0)
    optparams = OptimizerParams(; iterations_limit=10, trace_length=50, time_limit)
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    state = initial_state(o, x, pb; γ)
    xfinal_localNewton, tr = NSS.optimize!(pb, o, x; state, optparams)
    optimdata[o] = tr

    ## Build figures
    xopt = [1.0565430677054422, 0.0, 1.0846386699660968, 0.0, 2.8697105434169496, -0.9251005168121031, 1.2650147585707252, 0.0, 0.0, 0.0, 1.88295175081621, -0.2100432683452985, 0.07366979866412658, 0.0, 0.0, 1.2723048551572342, 0.39157637970150744, 2.304740111059459, 0.0, 0.0]
    Mopt = nothing
    Fopt = prevfloat(F(pb, xopt))

    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "nonconvex-MCP"; NUMEXPS_OUTDIR, plotgamma = false, includelegend=false)
    return true
end
