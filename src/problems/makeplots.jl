function buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, pbname::String; NUMEXPS_OUTDIR)
    @info "building figures for $pbname"

    stepinfo = treatproxsteps(pb, tr, Mopt, xopt)

    optimdatagamma = OrderedDict(
        L"\gamma low(x_k)" =>
            [(itstepinfo.γlow, itstepinfo.distopt) for itstepinfo in stepinfo],
        L"\bar{\gamma}(x_k)" =>
            [(itstepinfo.γup, itstepinfo.distopt) for itstepinfo in stepinfo],
        L"\gamma_k" => [(itstepinfo.γₖ, itstepinfo.distopt) for itstepinfo in stepinfo],
    )
    display(optimdatagamma)
    getabsc_distopt(o, trace) = [o[2] for o in trace]
    getord_gamma(o, trace) = [o[1] for o in trace]
    fig = plot_curves(
        optimdatagamma,
        getabsc_distopt,
        getord_gamma;
        xlabel=L"\| x_k - x^\star\|",
        ylabel=L"",
        # nmarks = 1000,
        xmode="log",
        includelegend = false,
    )
    savefig(fig, joinpath(NUMEXPS_OUTDIR, pbname*"_gamma"))

    # Suboptimality
    getabsc_time(optimizer, trace) = [state.time for state in trace]
    getord_subopt(optimizer, trace) = [state.Fx - Fopt for state in trace]
    fig = plot_curves(
        optimdata,
        getabsc_time,
        getord_subopt;
        xlabel="time (s)",
        ylabel=L"F(x_k) - F^\star",
        nmarks=1000,
        includelegend = false,
    )
    savefig(fig, joinpath(NUMEXPS_OUTDIR, pbname*"_time_subopt"))
end
