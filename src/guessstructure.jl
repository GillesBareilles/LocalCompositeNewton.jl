function areequal(M::EigmaxManifold, N::EigmaxManifold)
    return M.eigmult.r == N.eigmult.r
end
function areequal(M::MaxQuadManifold, N::MaxQuadManifold)
    return M.active_fᵢ_indices == N.active_fᵢ_indices
end
function areequal(M::FixedSparsityManifold, N::FixedSparsityManifold)
    return M.nz_coords == N.nz_coords
end

function guessstruct_prox(
    pb::NSP.Eigmax, firstorderderivatives::FirstOrderDerivativeInfo, γ
)
    _, active_indices = prox_max(firstorderderivatives.eigvals, γ)
    r = length(active_indices)

    # Codimension condition
    n = length(firstorderderivatives.x)
    rmax = 0
    while rmax * (rmax + 1) / 2 - 1 ≤ n
        rmax += 1
    end
    rmax -= 1

    return EigmaxManifold(pb, min(rmax, r))
end

function guessstruct_prox(
    pb::NSP.MaxQuadPb, firstorderderivatives::FirstOrderDerivativeInfo, γ
)
    _, active_indices = prox_max(firstorderderivatives.gx, γ)
    return NSP.MaxQuadManifold(pb, active_indices)
end

function guessstruct_prox(
    pb::NSP.MCPLeastSquares, firstorderderivatives::FirstOrderDerivativeInfo, γ
)
    n = pb.n
    resprox = zeros(n)
    NSP.proxr!(resprox, pb, firstorderderivatives.gx, γ) # NOTE: gx == x
    nz_coords = BitVector(map(t -> t!=0, resprox))

    return NSP.FixedSparsityManifold(pb, nz_coords)
end
