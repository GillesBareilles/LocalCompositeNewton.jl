get_legendname(obj::NSBFGS) = "nsBFGS"
get_legendname(obj::GradientSampling) = "GradientSampling"
get_legendname(obj::LocalCompositeNewtonOpt) = "LocalNewton"

getabsc_time(optimizer, trace) = [state.time for state in trace]
# getord_subopt(Fopt) = (optimizer, trace) -> [state.Fx-Fopt for state in trace]

# @noinline function plot_subopt_time(optimdata; Fopt, callback! = _->nothing)
#     fig = plot_curves(optimdata, getabsc_time, getord_subopt(Fopt);
#                       xlabel = "time (s)",
#                       ylabel = L"F(x_k) - F^\star",
#                       nmarks = 1000,
#                       callback!)
# end

# function runnumexps()
#     runexp_halfhalf()
#     runexp_maxquadBGLS()
#     runexp_eigmaxAL()
#     return
# end
