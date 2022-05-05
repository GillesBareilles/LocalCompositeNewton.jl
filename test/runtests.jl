using Test

include("SQP.jl")
include("oracles_eigmax.jl")

@testset "Experiment functions" begin
    @test LocalCompositeNewton.expe_maxquad()
    @test LocalCompositeNewton.expe_eigmax()
end
