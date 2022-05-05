using Test
using LocalCompositeNewton
using LinearAlgebra

@testset "get_lambda" begin
    m, n = 5, 10
    @testset "Tf = $Tf" for Tf in [Float64, BigFloat]
        @testset "point $i" for i in 1:10
            Jacₕ = rand(Tf, m, n)
            d = rand(Tf, n)

            wref = pinv(Jacₕ') * d
            w = LocalCompositeNewton.get_lambda(Jacₕ, d)

            @test w ≈ wref
            @test norm(Jacₕ * (d - Jacₕ' * w)) < 1e2 * eps(Tf)
        end
    end
end
