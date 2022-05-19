const ED = EigenDerivatives

function oracles_firstorder!(di::FirstOrderDerivativeInfo{Tf}, pb::Eigmax{Tf}, x) where Tf
    di.x .= x
    di.gx .= g(pb, x)
    di.eigvals, di.eigvecs = eigen(di.gx)
    di.Fx = maximum(di.eigvals)
    return nothing
end

function ∇²ϕᵢⱼ!(res::Matrix{Tf}, map, E, λs, τ, x, i, j, m, r) where {Tf}
    n = length(x)
    res .= 0

    for k in axes(res, 1), l in axes(res, 2)
        η = zeros(Tf, n)
        η[k] = 1

        res[k, l] = E[:, i]' * ED.D²g_kl(map, x, k, l) * E[:, j]
        for s in (r + 1):m
            scalar = 0.5 * (1 / (λs[i] - λs[s]) + 1 / (λs[j] - λs[s]))
            res[k, l] += scalar * (τ[i, s, k] * τ[j, s, l] + τ[i, s, l] * τ[j, s, k])
        end
    end
    return nothing
end

"""
    $TYPEDSIGNATURES


julia> using NonSmoothProblems, NonSmoothSolvers
julia> using BenchmarkTools
julia> using LocalCompositeNewton
julia> n = 20
julia> m = 25
julia> pb = get_eigmax_affine(; m, n, seed = 1864);
julia> M = EigmaxManifold(pb, 4)
julia> x = [-0.2903621117378083, -0.5906936043347821, 0.022580009190398934, -0.036962726426832536, -0.05311102380035424, -0.29609094669375197, 0.34511861687804846, 0.2917244946010309, 0.126235568114234, -0.2749392842386973, -0.7340388824944295, 0.5802709786578186, -0.23976054387022805, 0.4780418928792374, -0.4803760994489224, 0.1340455884010646, -0.009960099075968799, -0.8955982952380783, 0.03225246196467059, 0.16501043656163675];
julia> @benchmark LocalCompositeNewton.oracles!(di, pb, M, x) setup=(di = LocalCompositeNewton.DerivativeInfo(M, x))

BenchmarkTools.Trial: 367 samples with 1 evaluation.
 Range (min … max):  11.299 ms … 25.660 ms  ┊ GC (min … max): 0.00% … 34.54%
 Time  (median):     12.612 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   13.635 ms ±  3.032 ms  ┊ GC (mean ± σ):  5.20% ± 10.89%

  ▇█▅▄▂ ▄▁▁  ▄▆▂▂▂▁                                ▁
  █████▇████▇██████▆▄▁▇▁▄▄▄▄▆▁▄▄▆▇▄▁▄▁▄▁▆▁▁▁▁▄▁▁▁▄▄█▆▇▆▇▄▆▆▄▆ ▇
  11.3 ms      Histogram: log(frequency) by time      23.3 ms <

 Memory estimate: 6.07 MiB, allocs estimate: 41129.
"""
function oracles_structure!(
    di_struct::StructDerivativeInfo{Tf},
    di_fo::FirstOrderDerivativeInfo{Tf},
    pb::NonSmoothProblems.Eigmax{Tf,EigenDerivatives.AffineMap{Tf}},
    M,
    x::Vector{Tf},
) where {Tf}
    map = pb.A
    eigmult = M.eigmult
    r = eigmult.r

    gx = di_fo.gx
    λs, E = di_fo.eigvals, di_fo.eigvecs

    reverse!(λs)
    reverse!(E; dims=2)

    # Update ref point
    eigmult.x̄ .= x
    eigmult.Ē .= E[:, 1:(eigmult.r)]
    U = eigmult.Ē
    di_struct.x .= x

    hmat = U' * gx * U
    ED.hmat2vecsmall!(di_struct.hx, hmat, eigmult.r)
    # h!(di_struct.hx, eigmult, x, gx)

    # Update Jacobian
    for i in axes(di_struct.Jacₕ, 2)
        Dhmat = U' * ED.Dg_l(map, x, i) * U
        t = @view di_struct.Jacₕ[:, i]
        ED.hmat2vecsmall!(t, Dhmat, r)
    end

    # Update current gradient
    di_struct.∇Fx .= 0
    for l in axes(di_struct.∇Fx, 1), i in 1:(eigmult.r)
        di_struct.∇Fx[l] += U[:, i]' * ED.Dg_l(map, x, l) * U[:, i]
    end
    di_struct.∇Fx ./= eigmult.r

    # Update multiplier
    if length(di_struct.λ) > 0
        di_struct.λ .= get_lambda(di_struct.Jacₕ, di_struct.∇Fx)
    end

    # Update Lagrangian hessian
    di_struct.∇²Lx .= 0
    trλmult = sum(di_struct.λ[ED.l_partialdiag(r)])

    m = size(gx, 1)
    n = length(x)
    τ = zeros(Tf, r, m, n)
    # NOTE: this is the costliest part of the function.
    for i in 1:r, s in 1:m
        for k in 1:n
            τ[i, s, k] = E[:, i]' * ED.Dg_l(map, x, k) * E[:, s]
        end
    end

    temp = zeros(Tf, n, n)
    for l in ED.l_lowerdiag(r)
        i, j = ED.l2ij(l, r)
        λᵢⱼ = di_struct.λ[l]

        # ∇²ϕᵢⱼ!(temp, i, j)
        ∇²ϕᵢⱼ!(temp, map, E, λs, τ, x, i, j, m, r)
        di_struct.∇²Lx .+= -λᵢⱼ * temp
    end
    for l in ED.l_partialdiag(r)
        i, j = ED.l2ij(l, r)
        λᵢᵢ = di_struct.λ[l]

        # ∇²ϕᵢⱼ!(temp, i, j)
        ∇²ϕᵢⱼ!(temp, map, E, λs, τ, x, i, j, m, r)
        di_struct.∇²Lx .+= (1 / r - λᵢᵢ) * temp
    end
    i = j = r

    # ∇²ϕᵢⱼ!(temp, i, j)
    ∇²ϕᵢⱼ!(temp, map, E, λs, τ, x, i, j, m, r)
    di_struct.∇²Lx .+= (1 / r + trλmult) * temp

    return nothing
end
