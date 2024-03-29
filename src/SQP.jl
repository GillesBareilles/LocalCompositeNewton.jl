function get_SQP_direction_CG(
    pb, M, x::Vector{Tf}, structderivative; info=Dict()
) where {Tf}
    Z = nullspace(structderivative.Jacₕ)         # tangent space basis

    ## 1. Restoration step
    r = zeros(Tf, size(x))
    if norm(structderivative.hx) > 1e1 * eps(Tf)
        r = IterativeSolvers.lsmr(structderivative.Jacₕ, -structderivative.hx)
    end

    ## 2. Reduced gradient and RHS
    g = Z' * structderivative.∇Fx
    v = -g - Z' * structderivative.∇²Lx * r

    ## 3. Linear system solve
    u = (Z' * structderivative.∇²Lx * Z) \ v

    d = r + Z * u
    return d
end

# function get_SQP_direction_JuMP(pb, M, x::Vector; regularize=false)
#     @assert manifold_codim(M) > 0

#     ## Oracle calls
#     n = length(x)
#     hx, Jacₕ, ∇Fx, λ, ∇²Lx = oracles(pb, x, M)

#     model = Model(optimizer_with_attributes(OSQP.Optimizer, "polish" => true))
#     set_silent(model)

#     d = @variable(model, d[1:n])
#     @constraint(model, hx + Jacₕ * d .== 0)

#     if regularize
#         gradFx = ∇Fx - Jacₕ' * λ
#         ∇²Lx += ((norm(gradFx))^(0.8))I
#     end

#     @objective(model, Min, ∇Fx' * d + 0.5 * d' * ∇²Lx * d)

#     # @info "Solving for SQP step"
#     JuMP.optimize!(model)
#     d = value.(d)
#     if termination_status(model) != MathOptInterface.OPTIMAL
#         @warn "Problem in SQP direction computation" termination_status(model) primal_status(
#             model
#         ) dual_status(model)
#         d .= 0
#     end

#     return d, Jacₕ
# end

function addMaratoscorrection!(d::Vector{Tf}, pb, M, x, Jacₕ) where {Tf}
    hxd = zeros(Tf, size(Jacₕ, 1))
    if isa(pb, Eigmax)
        h!(hxd, M.eigmult, x .+ d, g(pb, x .+ d))
    else
        hxd = NSP.h(M, x .+ d)
    end
    norm(hxd) < 1e1 * eps(Tf) && return
    dMaratos = IterativeSolvers.lsmr(Jacₕ, -hxd)
    @debug "Maratos SOC: " norm(hx + Jacₕ * dMaratos)
    d .+= dMaratos
    return
end
