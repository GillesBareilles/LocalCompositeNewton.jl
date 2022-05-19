# Local Newton Method
struct LocalCompositeNewtonOpt{Tf} <: NSS.NonSmoothOptimizer{Tf}
    start_it::Int64
    start_time::Float64
end

Base.@kwdef mutable struct LocalCompositeNewtonState{Tf,Tm} <: NSS.OptimizerState{Tf}
    x::Vector{Tf} # point
    it::Int64     # iteration
    M::Tm         # current manifold
    γ::Tf         # current step
    di_fo::FirstOrderDerivativeInfo{Tf}
    di_fonext::FirstOrderDerivativeInfo{Tf}
    di_struct::StructDerivativeInfo{Tf}
end

function initial_state(o::LocalCompositeNewtonOpt, xinit::Vector{Tf}, pb; γ=100.0) where {Tf}
    Minit = NSP.point_manifold(pb, xinit)
    state = LocalCompositeNewtonState(;
        x         = xinit,
        it        = o.start_it,
        M         = NSP.point_manifold(pb, xinit),
        γ         = Tf(γ),
        di_fo     = FirstOrderDerivativeInfo(pb, xinit),
        di_fonext = FirstOrderDerivativeInfo(pb, xinit),
        di_struct = StructDerivativeInfo(Minit, xinit),
    )
    oracles_firstorder!(state.di_fo, pb, xinit)
    return state
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

function update_iterate!(state, ::LocalCompositeNewtonOpt{Tf}, pb) where {Tf}
    x = state.x
    Fx = state.di_fo.Fx

    ## Identification
    state.γ /= 2
    M = guessstruct_prox(pb, state.di_fo, state.γ)

    if !areequal(M, state.M)
        @info "changing manifolds" M state.M
        state.di_struct = StructDerivativeInfo(M, x)
    end

    ## Step
    oracles_structure!(state.di_struct, state.di_fo, pb, M, x)

    info = Dict()
    d = get_SQP_direction_CG(pb, M, x, state.di_struct; info)

    # @warn "No Maratos"
    addMaratoscorrection!(d, pb, M, x, state.di_struct.Jacₕ)

    oracles_firstorder!(state.di_fonext, pb, state.x+d)
    Fxd = state.di_fonext.Fx
    if Fxd < Fx
        state.x .+= d

        update_difirstorder!(state)
        state.di_fonext = state.di_fo
    else
        @warn "not changing point" Fxd Fx
    end
    state.M = M

    return (; :normd => norm(d), :M => M),
    norm(d) < 10 * eps(Tf) ? NSS.problem_solved : NSS.iteration_completed
end
