# function guess_prox_intermediatespace(pb::NSP.Eigmax, x, γ)
#     _, active_indices = prox_max(eigvals(NSP.g(pb, x)), γ)
#     return NSP.EigmaxManifold(pb, pb.m - minimum(active_indices) + 1)
# end
function guessstruct_prox(pb::NSP.Eigmax, x, γ)
    Λ = eigvals(NSP.g(pb, x))
    _, active_indices = prox_max(Λ, γ)
    r = length(active_indices)

    # Codimension condition
    rmax = 0
    while rmax * (rmax + 1) / 2 - 1 ≤ length(x)
        rmax += 1
    end
    rmax -= 1

    return EigmaxManifold(pb, min(rmax, r))
end
function guessstruct_prox(pb::NSP.MaxQuadPb, x, γ)
    _, active_indices = prox_max(NSP.g(pb, x), γ)
    return NSP.MaxQuadManifold(pb, active_indices)
end

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

function areequal(M::EigmaxManifold, N::EigmaxManifold)
    return M.eigmult.r == N.eigmult.r
end
function areequal(M::MaxQuadManifold, N::MaxQuadManifold)
    return M.active_fᵢ_indices == N.active_fᵢ_indices
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
    # regularize = false
    # d, Jacₕ = get_SQP_direction_JuMP(pb, M, x; regularize)
    info = Dict()
    d = get_SQP_direction_CG(pb, M, x, state.di; info)

    @warn "No Maratos"
    # fixMaratos = true
    # fixMaratos = false
    # fixMaratos && addMaratoscorrection!(d, pb, M, x, Jacₕ)

    if F(pb, x + d) < F(pb, x)
        state.x .+= d
    end
    state.M = M

    return (; :normd => norm(d), :M => M),
    norm(d) < 5e-14 ? NSS.problem_solved : NSS.iteration_completed
end
