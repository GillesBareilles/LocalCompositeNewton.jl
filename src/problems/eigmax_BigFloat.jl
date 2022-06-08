function expe_eigmax_BigFloat(NUMEXPS_OUTDIR=NUMEXPS_OUTDIR_DEFAULT)
    Tf = BigFloat
    n = 25
    m = 50
    pb = get_eigmax_affine(; m, n, seed=1864, Tf)

    # Initial point is perturbed minimum
    x = BigFloat[
        0.2272559623994859039900704361320364123810869804733328004919319321531380662423105,
        0.5309138304660964743682160227943703375287200438755596757251646756264926679894523,
        0.0261584590702443896995413472868437028502199010687238826996586675272233194838452,
        0.07968822567879124795107229231862416888061064960937965960160790584206274016598455,
        -0.1351463843097272182846165004910713045758580315350179187340070866704335017711361,
        0.020977360999724221653610514607038516134074971299809321252680551287543774916397,
        0.009856157326881099122314179417944996928638502641856048841022553969649365466222093,
        -0.08116587525168686240256271072283577840592014892676428895345273621758111613042423,
        -0.2392351486907360736504170811229296245454350578444132166374241548946909403809367,
        -0.639603796201215803357555295064838028608415801112786230035376713146569238722033,
        -0.01819281819326752589951249334451725291802066978972580014651346772398102698690926,
        0.009978125943603714905335079750405149546035339784929975459643696956568604032009636,
        0.2668379454984990484985121952998384598020908860676752875750954939616809290473602,
        -0.05436964905494112143780511884717277883131708177228193213919203251032931858379721,
        -0.1939000876424110566589945278796221473188167797815923478773325815697469698111774,
        -0.05629877360058244122500423998564533408890238826479651114695393937417726250259303,
        0.1293905568689126352325686466835841519672988943703635061704908019939606380528242,
        0.1357124305212498377384581587776170146228193622177478201476427417323626218062517,
        -0.1537122150895015093554987629621704527449038687834824376189599369692378627711753,
        -0.8224263652185227325646208109410651764328191659414751985005916992919342485433642,
        -0.1493704035040118884543925556838979372202717681089608246684696207044702918703155,
        0.1879925066980093956000032499601084118325426189352086567435208060268432454242027,
        -0.5431239914705633992776929974130251223647887292017617970080322180965401301518494,
        -0.29948059195742770214435443791054224616210310828827667105505773883309259768303,
        0.07344598165996554743788936098644495192850487381105268958232068083551512427203618,
    ]
    x += 1e-4 * ones(n)

    optparams_precomp = OptimizerParams(;
        iterations_limit=2, trace_length=0, time_limit=0.5
    )
    time_limit = 60

    ## Define solvers to be run on problem
    optimdata = OrderedDict()

    # Gradient sampling
    o = GradientSampling(x)
    optparams = OptimizerParams(;
        iterations_limit=100, trace_length=100, time_limit=time_limit
    )
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_gs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # nsBFGS
    o = NSBFGS{Tf}()
    optparams = OptimizerParams(;
        iterations_limit=300, trace_length=100, time_limit=time_limit
    )
    _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    xfinal_nsbfgs, tr = NSS.optimize!(pb, o, x; optparams)
    optimdata[o] = tr

    # Local Newton method
    getx(o, os) = deepcopy(os.x)
    getγ(o, os) = deepcopy(os.γ)
    optimstate_extensions = OrderedDict{Symbol,Function}(:x => getx, :γ => getγ)

    # find the smaller γ which gives maximal structure
    # Note that the maximum structure here is r=6 and not b=25.
    # Indeed, the codimension of the structure manifold exceeds n=25 for r>6.
    gx = eigvals(g(pb, x))
    γ = 0.0
    for i in 1:6
        @show (gx[end - i + 1] - gx[end - i]) * i
        γ += (gx[end - i + 1] - gx[end - i]) * i
    end
    # @show gx[end - 4:end]
    # @show guessstruct_prox(pb, x, γ)
    # @show guessstruct_prox(pb, x, γ - 0.01)
    # @show guessstruct_prox(pb, x, γ)
    # @show guessstruct_prox(pb, x, γ + 0.01)
    # @show guessstruct_prox(pb, x, γ / 10)
    # @show γ
    # γ *= 2

    o = LocalCompositeNewtonOpt{Tf}(0, 0.0)
    optparams = OptimizerParams(;
        iterations_limit=10, trace_length=50, time_limit=time_limit
    )
    # _ = NSS.optimize!(pb, o, x; optparams=optparams_precomp)
    state = initial_state(o, x, pb; γ)
    xfinal_localNewton, tr = NSS.optimize!(
        pb, o, x; state, optparams, optimstate_extensions
    )
    optimdata[o] = tr

    ## Build figures
    xopt = xfinal_localNewton
    Mopt = EigmaxManifold(pb, 3)
    Fopt = prevfloat(F(pb, xopt))

    buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, "eigmax_BigFloat"; NUMEXPS_OUTDIR)

    return true
end
