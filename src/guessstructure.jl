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

function areequal(M::EigmaxManifold, N::EigmaxManifold)
    return M.eigmult.r == N.eigmult.r
end
function areequal(M::MaxQuadManifold, N::MaxQuadManifold)
    return M.active_fᵢ_indices == N.active_fᵢ_indices
end
