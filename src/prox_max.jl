"""
    prox_max(x, γ)

Compute the proximal operator of function `γ max` at point `x`, returns its
output and the vector of active indices.
"""
function prox_max(x, γ)
    p = sortperm(x, rev=true)
    xsort = @view x[p]
    res = copy(x)

    # Numberof active coordinates of x
    nb_act = 1
    γremaining = γ
    while true
        if γremaining < nb_act * (xsort[nb_act] - xsort[nb_act+1])
            res[p[1:nb_act]] .= xsort[nb_act+1] + (xsort[nb_act] - xsort[nb_act+1]) - γremaining / nb_act
            break
        else
            γremaining -= nb_act * (xsort[nb_act] - xsort[nb_act+1])
            nb_act += 1
        end

        if nb_act == length(x)
            res .= xsort[end] - γremaining / nb_act
            break
        end
    end


    ## Checking correctness
    if !(res[p[nb_act+1:end]] == x[p[nb_act+1:end]]) || !(sum(x[p[i]] - res[p[i]] for i in 1:nb_act) ≈ γ)
        # @show (res[p[nb_act+1:end]] == x[p[nb_act+1:end]])
        # @show sum(x[p[i]] - res[p[i]] for i in 1:nb_act), γ
        @error "ProxMax: prox computation doesn't check out."
    end

    structure = p[1:nb_act]

    return res, structure
end
