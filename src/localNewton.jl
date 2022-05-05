# Local Newton Method
struct LocalCompositeNewtonOpt{Tf} <: NSS.NonSmoothOptimizer{Tf}
    start_it::Int64
    start_time::Float64
end

Base.@kwdef mutable struct LocalCompositeNewtonState{Tf,Tm} <: NSS.OptimizerState{Tf}
    x::Vector{Tf}   # point
    it::Int64       # iteration
    M::Tm           # current manifold
    γ::Float64      # current step
    di::DerivativeInfo{Tf}
end

function initial_state(o::LocalCompositeNewtonOpt, xinit, pb)
    Minit = NSP.point_manifold(pb, xinit)
    return LocalCompositeNewtonState(;
        x=xinit,
        it=o.start_it,
        M=NSP.point_manifold(pb, xinit),
        γ=100.0,
        di=DerivativeInfo(Minit, xinit),
    )
end

#
### Printing
#
print_header(::LocalCompositeNewtonOpt) = println("**** LocalCompositeNewtonOpt algorithm")
get_minimizer_candidate(state::LocalCompositeNewtonState) = state.x

display_logs_header_post(::LocalCompositeNewtonOpt) = print("|dSQP|      M")
function display_logs_post(os, ::LocalCompositeNewtonOpt)
    @printf "%.3e   %s" os.additionalinfo.normd os.additionalinfo.M
end

function update_iterate!(state, ::LocalCompositeNewtonOpt, pb)
    x = state.x

    ## Identification
    state.γ /= 10
    M = guessstruct_prox(pb, x, state.γ)

    if !areequal(M, state.M)
        @info "changing manifolds" M state.M
        state.di = DerivativeInfo(M, x)
    end

    ## Step
    # d, Jacₕ = get_SQP_direction_JuMP(pb, M, x; regularize = false)
    info = Dict()
    d = get_SQP_direction_CG(pb, M, x, state.di; info)

    @warn "No Maratos"
    # addMaratoscorrection!(d, pb, M, x, Jacₕ)

    if F(pb, x + d) < F(pb, x)
        state.x .+= d
    end
    state.M = M

    return (; :normd => norm(d), :M => M),
    norm(d) < 5e-14 ? NSS.problem_solved : NSS.iteration_completed
end
