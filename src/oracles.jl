mutable struct FirstOrderDerivativeInfo{Tf,Tgx}
    x::Vector{Tf}
    Fx::Tf
    gx::Tgx
    eigvals::Vector{Tf}
    eigvecs::Matrix{Tf}
end
function FirstOrderDerivativeInfo(pb, x::Vector{Tf}) where {Tf}
    n = length(x)
    gx = g(pb, x)
    Tgx = typeof(gx)
    return FirstOrderDerivativeInfo{Tf, Tgx}(
        zeros(Tf, n), Tf(0), similar(gx), zeros(Tf, size(gx, 1)), zeros(Tf, n, n)
    )
end

"""
    $TYPEDSIGNATURES

Replace the information in `state.di_fo` by that in `state.di_fonext`.
Happens when a point is updated.
"""
function update_difirstorder!(state)
    state.di_fo.x .= state.di_fonext.x
    state.di_fo.Fx = state.di_fonext.Fx
    state.di_fo.gx .= state.di_fonext.gx
    state.di_fo.eigvals .= state.di_fonext.eigvals
    state.di_fo.eigvecs .= state.di_fonext.eigvecs
    return nothing
end

mutable struct StructDerivativeInfo{Tf}
    x::Vector{Tf}
    hx::Vector{Tf}
    λ::Vector{Tf}
    Jacₕ::Matrix{Tf}
    ∇Fx::Vector{Tf}
    ∇²Lx::Matrix{Tf}
end
function StructDerivativeInfo(M, x::Vector{Tf}) where {Tf}
    n = length(x)
    p = manifold_codim(M)
    return StructDerivativeInfo(
        zeros(Tf, n),
        zeros(Tf, p),
        zeros(Tf, p),
        zeros(Tf, p, n),
        zeros(Tf, n),
        zeros(Tf, n, n),
    )
end

function oracles_firstorder!(di::FirstOrderDerivativeInfo{Tf}, pb, x) where {Tf}
    @info "calling generic firstorder oracle"
    di.x .= x
    di.gx .= g(pb, x)
    return di.Fx = F(pb, x)
end

function oracles_structure!(
    di::StructDerivativeInfo{Tf}, ::FirstOrderDerivativeInfo{Tf}, pb, M, x::Vector{Tf}
) where {Tf}
    @info "calling generic structure oracle"
    di.x .= x

    di.hx .= NSP.h(M, x)
    di.Jacₕ .= NSP.Jac_h(M, x)
    di.∇Fx .= NSP.∇F̃(pb, M, x)
    di.λ .= get_lambda(di.Jacₕ, di.∇Fx)

    di.∇²Lx .= NSP.∇²L(pb, M, x, di.λ)
    return nothing
end

function get_lambda(Jacₕ::Matrix{Tf}, d::Vector{Tf}) where {Tf}
    @debug "rank should be maximal for quadratic SQP rate" rank(Jacₕ) size(Jacₕ)

    Q, R = qr(Jacₕ)
    w = Q * ((R * R') \ (R * d))
    return w
end
