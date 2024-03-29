module LocalCompositeNewton

# using MathOptInterface
# using JuMP
# using OSQP
using IterativeSolvers

using LinearAlgebra
using Printf
using DataStructures
using Random
using EigenDerivatives
using NonSmoothProblems
using NonSmoothSolvers

import NonSmoothSolvers:
    initial_state,
    print_header,
    display_logs_header_post,
    display_logs_post,
    update_iterate!,
    get_minimizer_candidate,
    has_converged

using PlotsOptim
import PlotsOptim: get_legendname
using LaTeXStrings
using DocStringExtensions

# Setting numerical experiments default output directory
const NUMEXPS_OUTDIR_DEFAULT = joinpath(
    dirname(pathof(LocalCompositeNewton)), "..", "numexps_output"
)
function __init__()
    if !isdir(NUMEXPS_OUTDIR_DEFAULT)
        mkdir(NUMEXPS_OUTDIR_DEFAULT)
    end
    @info "default output directory for numerical experiments is: " NUMEXPS_OUTDIR_DEFAULT
    return nothing
end

include("prox_max.jl")
include("oracles.jl")
include("oracle_maxquad.jl")
include("oracle_eigmax.jl")
include("oracle_MCPLeastSquares.jl")
include("guessstructure.jl")

include("SQP.jl")
include("localNewton.jl")

# Float64 experiments
include("problems/maxquadBGLS.jl")
include("problems/eigmax.jl")
# BigFloat experiments
include("problems/maxquadBGLS_BigFloat.jl")
include("problems/eigmax_BigFloat.jl")
# Nonconvex experiments
include("problems/nonconvex_maxquad.jl")
include("problems/nonconvex_MCP.jl")


include("makeplots.jl")

export optimize!

end # module
