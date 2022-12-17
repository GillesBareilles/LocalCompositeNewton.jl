get_legendname(obj::NSBFGS) = "nsBFGS"
get_legendname(obj::GradientSampling) = "GradientSampling"
get_legendname(obj::LocalCompositeNewtonOpt) = "LocalNewton"

getabsc_time(optimizer, trace) = [state.time for state in trace]

function runnumexps()
    expe_maxquad()
    expe_eigmax()
    return nothing
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
        push!(stepinfo, (; γlow, γup, γₖ, distopt))
    end
    return stepinfo
end
function treatproxsteps(pb, tr, Mopt::MaxQuadManifold, xopt::Vector{Tf}) where {Tf}
    stepinfo = Any[]
    for os in tr[1:end]
        x = os.additionalinfo.x
        gx = NSP.g(pb, x)

        # Computing steps low, up
        r = length(Mopt.active_fᵢ_indices)
        γlow, γup = get_γlowγupmax(gx, r)
        γₖ = os.additionalinfo.γ
        distopt = norm(x - xopt)
        distopt == 0 && (distopt = eps(Tf))
        push!(stepinfo, (; γlow, γup, γₖ, distopt))
    end
    return stepinfo
end
function get_γlowγupmax(gx, r)
    gxsort = sort(gx; rev=true)

    γlow = 0
    for k in 1:(r - 1)
        γlow += k * (gxsort[k] - gxsort[k + 1])
    end
    γup = γlow + r * (gxsort[r] - gxsort[r + 1])

    return γlow, γup
end

function buildfigures(optimdata, tr, pb, xopt, Mopt, Fopt, pbname::String; NUMEXPS_OUTDIR, plotgamma = true, includelegend = false)
    @info "building figures for $pbname"

    if plotgamma
        stepinfo = treatproxsteps(pb, tr, Mopt, xopt)

        optimdatagamma = OrderedDict(
            L"\gamma low(x_k)" =>
                [(itstepinfo.γlow, itstepinfo.distopt) for itstepinfo in stepinfo],
            L"\bar{\gamma}(x_k)" =>
                [(itstepinfo.γup, itstepinfo.distopt) for itstepinfo in stepinfo],
            L"\gamma_k" => [(itstepinfo.γₖ, itstepinfo.distopt) for itstepinfo in stepinfo],
        )

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
            includelegend=false,
        )
        savefig(fig, joinpath(NUMEXPS_OUTDIR, pbname * "_gamma"))
    end

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
        includelegend,
    )
    return savefig(fig, joinpath(NUMEXPS_OUTDIR, pbname * "_time_subopt"))
end
