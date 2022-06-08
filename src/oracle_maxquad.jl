function oracles_firstorder!(
    di::FirstOrderDerivativeInfo{Tf}, pb::MaxQuadPb{Tf}, x
) where {Tf}
    di.x .= x
    di.gx .= g(pb, x)
    di.Fx = maximum(di.gx)
    return nothing
end

function oracles_structure!(
    di::StructDerivativeInfo{Tf},
    ::FirstOrderDerivativeInfo{Tf},
    pb::MaxQuadPb{Tf},
    M,
    x::Vector{Tf},
) where {Tf}
    di.x .= x
    p = manifold_codim(M)
    m = p + 1

    actindsbutone = @view M.active_fᵢ_indices[1:(end - 1)]
    actindlast = M.active_fᵢ_indices[end]

    # h function
    for (i, ind) in enumerate(actindsbutone)
        di.hx[i] = NSP.gᵢ(pb, x, ind)
    end
    di.hx .-= NSP.gᵢ(pb, x, actindlast)

    # h Jacobian
    lastgrad = NSP.∇gᵢ(pb, x, actindlast)
    for (i, ind) in enumerate(actindsbutone)
        di.Jacₕ[i, :] .= NSP.∇gᵢ(pb, x, ind) .- lastgrad
    end

    # Smooth extension gradient, normal coordinates
    di.∇Fx .= lastgrad
    di.λ .= get_lambda(di.Jacₕ, di.∇Fx)

    # Lagrangian hessian
    di.∇²Lx .= 0
    for (i, ind) in enumerate(actindsbutone)
        di.∇²Lx .+= -di.λ[i] .* NSP.∇²gᵢ(pb, x, ind)
    end
    di.∇²Lx .+= (1 + sum(di.λ)) .* NSP.∇²gᵢ(pb, x, actindlast)
    return nothing
end
