struct DerivativeInfo{Tf}
    x::Vector{Tf}
    hx::Vector{Tf}
    λ::Vector{Tf}
    Jacₕ::Matrix{Tf}
    ∇Fx::Vector{Tf}
    ∇²Lx::Matrix{Tf}
end
function DerivativeInfo(M, x::Vector{Tf}) where {Tf}
    n = length(x)
    p = manifold_codim(M)
    return DerivativeInfo(
        zeros(Tf, n),
        zeros(Tf, p),
        zeros(Tf, p),
        zeros(Tf, p, n),
        zeros(Tf, n),
        zeros(Tf, n, n),
    )
end

# function oracles!(di::DerivativeInfo{Tf}, pb, M, x::Vector{Tf}) where {Tf}
#     di.x .= x

#     di.hx .= NSP.h(M, x)
#     di.Jacₕ .= NSP.Jac_h(M, x)
#     di.∇Fx .= NSP.∇F̃(pb, M, x)
#     di.λ .= get_lambda(di.Jacₕ, di.∇Fx)

#     di.∇²Lx .= NSP.∇²L(pb, M, x, di.λ)
#     return nothing
# end

function get_lambda(Jacₕ::Matrix{Tf}, d::Vector{Tf}) where {Tf}
    @debug "rank should be maximal for quadratic SQP rate" rank(Jacₕ) size(Jacₕ)

    Q, R = qr(Jacₕ)
    w = Q * ((R * R') \ (R * d))
    return w
end
