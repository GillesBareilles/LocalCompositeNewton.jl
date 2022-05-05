using Test
using NonSmoothProblems
using LocalCompositeNewton
using EigenDerivatives

const LCN = LocalCompositeNewton
const ED = EigenDerivatives

@testset "Test eigmax oracles" begin
    @testset "Tf = $Tf" for Tf in [Float64, BigFloat]
        @testset "run $rnd" for rnd in 1:5
            n = 20
            m = 25
            pb = get_eigmax_affine(; m, n, seed=1864, Tf)
            x = rand(Tf, n)
            M = EigmaxManifold(pb, 4)

            di = LCN.DerivativeInfo(M, x)
            LCN.oracles!(di, pb, M, x)

            # Test relative to EigenDerivatives
            A = pb.A
            eigmult = M.eigmult
            ED.update_refpoint!(eigmult, A, x)
            gx = g(A, x)
            m = manifold_codim(M)

            @test di.x ≈ x

            hx = zeros(Tf, m)
            ED.h!(hx, eigmult, x, gx)
            @test di.hx ≈ hx

            Jacₕ = zeros(Tf, m, n)
            ED.Jacₕ!(Jacₕ, eigmult, pb.A, x)
            Jacₕ!(Jacₕ, eigmult, pb.A, x)
            @test di.Jacₕ ≈ Jacₕ

            ∇Fx = similar(x)
            ED.∇F̃!(∇Fx, eigmult, pb.A, x)
            @test di.∇Fx ≈ ∇Fx

            # Lambda is tested elsewhere

            ∇²Lx = zeros(Tf, n, n)
            ED.∇²L!(∇²Lx, eigmult, pb.A, x, di.λ, gx)
            @test ∇²Lx ≈ di.∇²Lx
        end
    end
end
