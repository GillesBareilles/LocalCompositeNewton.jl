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
            pb = get_eigmax_affine(; m, n, seed=1864 + rnd, Tf)
            x = rand(Tf, n)
            M = EigmaxManifold(pb, 4)

            eigmult = M.eigmult
            m = manifold_codim(M)
            A = pb.A

            # Test relative to EigenDerivatives
            di_fo = LCN.FirstOrderDerivativeInfo(pb, x)
            LCN.oracles_firstorder!(di_fo, pb, x)

            ED.update_refpoint!(eigmult, A, x)
            gx = g(A, x)
            @test di_fo.x ≈ x
            @test di_fo.gx ≈ gx

            di_struct = LCN.StructDerivativeInfo(M, x)
            LCN.oracles_structure!(di_struct, di_fo, pb, M, x)

            @test di_struct.x ≈ x

            hx = zeros(Tf, m)
            ED.h!(hx, eigmult, x, gx)
            @test di_struct.hx ≈ hx

            Jacₕ = zeros(Tf, m, n)
            ED.Jacₕ!(Jacₕ, eigmult, pb.A, x)
            Jacₕ!(Jacₕ, eigmult, pb.A, x)
            @test di_struct.Jacₕ ≈ Jacₕ

            ∇Fx = similar(x)
            ED.∇F̃!(∇Fx, eigmult, pb.A, x)
            @test di_struct.∇Fx ≈ ∇Fx

            # Lambda is tested elsewhere

            ∇²Lx = zeros(Tf, n, n)
            ED.∇²L!(∇²Lx, eigmult, pb.A, x, di_struct.λ, gx)
            @test ∇²Lx ≈ di_struct.∇²Lx
        end
    end
end
