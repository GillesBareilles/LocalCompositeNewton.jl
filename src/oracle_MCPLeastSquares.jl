function oracles_firstorder!(
    di::FirstOrderDerivativeInfo{Tf}, pb::MCPLeastSquares{Tf}, x
) where {Tf}
    di.x .= x
    di.gx .= x
    di.Fx = F(pb, x)
    return nothing
end

function oracles_structure!(
    di::StructDerivativeInfo{Tf},
    ::FirstOrderDerivativeInfo{Tf},
    pb::MCPLeastSquares{Tf},
    M,
    x::Vector{Tf},
) where {Tf}
    di.x .= x

    di.hx .= NSP.h(M, x)
    di.Jacₕ .= NSP.Jac_h(M, x)
    di.∇Fx .= NSP.∇F̃(pb, M, x)
    di.λ .= get_lambda(di.Jacₕ, di.∇Fx)

    # only specilize the lagrangian hessian
    di.∇²Lx .= pb.A' * pb.A
    for i in axes(x, 1)
        if abs(x[i]) ≤ pb.β * pb.λ
            di.∇²Lx[i, i] += - 1 / pb.β
        end
    end

    return nothing
end
