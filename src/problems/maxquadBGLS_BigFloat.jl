function expe_maxquad_BigFloat(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = BigFloat
    pb = MaxQuadBGLS(Tf)
    # x = zeros(Tf, 10) .+ 1.0

    x = BigFloat[
        0.1262565807610255445185179672585653584266799011761391068030613140161575044921407,
        0.03437830256097890704051252130186747265414216053032158387085577917658060836438285,
        0.00685719836779687427825948734161445053288243455223652272329090200614780597917616,
        -0.02636065820494096094609872850478543540617664857926117605680110401019568120571228,
        -0.06729492270868006593536380517000718723176441530271898158742984678710444333790462,
        0.2783995007541536495680073389703896098041667741271222704810012994401216165592615,
        -0.07421866452333002987978647728762983359129255712282503412231953553997394268371833,
        -0.1385240478296728974544121542500292231851791141986074340874637974995779468755006,
        -0.08403122315682629669364179873407886367397270685886949091426946545541491579349118,
        -0.03858030979552272014360079500149203641502234120798212060288844780395701918134798,
    ]
    x .+= 1e-2 .* ones(Tf, 10)

    optparams_precomp = OptimizerParams(;
        iterations_limit=2, trace_length=0, time_limit=0.5
    )

    ## Define solvers to be run on problem
    optimdata = OrderedDict()
    time_limit = 0.1

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

    buildfigures(
        optimdata, tr, pb, xopt, Mopt, Fopt, "maxquadBGLS_BigFloat"; NUMEXPS_OUTDIR
    )
    return true
end
