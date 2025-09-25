using StaticArrays
using LinearAlgebra, Statistics
using Test

using CompScienceMeshes
import CompScienceMeshes: chart, measure, jacobian, refnodes, neighborhood,
                          coordtype, dimension, universedimension, vertextype,
                          quadpoints

using BEAST
using FastGaussQuadrature
using SpecialFunctions
using Gmsh

using LinearAlgebra: norm

quad_measure(ch, n::Int=5) = begin
    q = CompScienceMeshes.quadpoints(ch, n)
    w = q isa Tuple{AbstractVector,AbstractVector} ? q[2] : last.(q)
    sum(w)
end


# ------------------------------------------------------------------------------
# 0) Tiny handmade CurvilinearMesh sanity (p=2 line cells in ℝ²)
# ------------------------------------------------------------------------------

# --- replace the handmade mesh block with this ---
v1 = SVector(1.0, 0.0)
v2 = SVector(0.0, 1.0)
v3 = SVector(-1.0, 0.0)
v4 = 0.5*(v1 + v2)     # midpoint for edge (1,2)
v5 = 0.5*(v2 + v3)     # midpoint for edge (2,3)

verts = [v1, v2, v3, v4, v5]
faces = [SVector(1,2,4),   # cell 1: (end1, end2, mid)
         SVector(2,3,5)]   # cell 2: (end1, end2, mid)
         
m0 = CompScienceMeshes.CurvilinearMesh(verts, faces, 2)

@testset "CurvilinearMesh basics (handmade)" begin
    @test CompScienceMeshes.mesh_order(m0) == 2
    @test CompScienceMeshes.dimension(m0) == 1
    @test CompScienceMeshes.universedimension(m0) == 2
    @test CompScienceMeshes.vertextype(m0) == SVector{2,Float64}
    @test CompScienceMeshes.coordtype(m0) == Float64
    
    ch = chart(m0, first(cells(m0)))
    ζ = refnodes(ch)
    @test length(ζ) == 3
    
    # nodal interpolation sanity (your CurvilinearSimplex stores field `vertices`, not `X`)
    @test map(ch, ζ[1]) == ch.vertices[1]
    @test map(ch, ζ[2]) == ch.vertices[2]
    @test map(ch, ζ[3]) == ch.vertices[3]
    
    ξ, w = (-sqrt(3/5), 0.0, sqrt(3/5)), (5/9, 8/9, 5/9)
    q = sum(w[k]*jacobian(ch, (ξ[k]+1)/2)*0.5 for k in 1:3)
    @test isapprox(measure(ch), q; atol=1e-14, rtol=0)
    
    qp = CompScienceMeshes.quadpoints(ch, 5)
    pts, w_phys = qp isa Tuple{AbstractVector,AbstractVector} ? qp : (first.(qp), last.(qp))
    
    meas_qp = sum(w_phys)  # weights already include J(ζ)
    @test isapprox(sum(w_phys), CompScienceMeshes.measure(ch); rtol=1e-13)
    @test all(ζ -> CompScienceMeshes.jacobian(ch, ζ) > 0, getindex.(CompScienceMeshes.parametric.(first.(quadpoints(ch,5))),1))
    @test isapprox(quad_measure(ch,5), CompScienceMeshes.measure(ch); rtol=1e-13)

end


# ------------------------------------------------------------------------------
# 1) Build a quadratic circle boundary with Gmsh and load as CurvilinearMesh
# ------------------------------------------------------------------------------

gmsh.initialize()
try
    gmsh.model.add("disk_p2")
    s_tag = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.10)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

    gmsh.model.addPhysicalGroup(2, [s_tag], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    dimtags = [(Int32(2), Int32(s_tag))]
    bnd = gmsh.model.getBoundary(dimtags, false, false, false)
    curves = [t[2] for t in bnd if t[1] == 1]
    gmsh.model.addPhysicalGroup(1, curves, 2)
    gmsh.model.setPhysicalName(1, 2, "Boundary")

    gmsh.model.mesh.generate(2)
    mshpath = joinpath(@__DIR__, "assets", "circle2d_quadratic.msh")
    isdir(dirname(mshpath)) || mkpath(dirname(mshpath))
    gmsh.write(mshpath)
finally
    gmsh.finalize()
end

m = load_gmsh_mesh(
    joinpath(@__DIR__, "assets", "circle2d_quadratic.msh");
    udim   = 2,
    element = :line,
    order   = 2,
    physical = "Boundary",
)

@testset "Loaded circle p=2 mesh: geometry sanity" begin
    @test typeof(m) <: CompScienceMeshes.CurvilinearMesh

    # perimeter ~ 2π
    r = 1.0
    L = sum(measure(chart(m, i)) for i in cells(m))
    @test isapprox(L, 2π*r; rtol=5e-3)

    # radii close to r at endpoints & midpoints
    rs = Float64[]
    for i in cells(m)
        ch = chart(m, i)
        for ζ in (0.0, 0.5, 1.0)
            push!(rs, norm(map(ch, ζ)))
        end
    end
    @test maximum(abs.(rs .- r)) < 5e-3

    # neighborhood API spot check
    ch = chart(m, first(cells(m)))
    ζ  = SVector{1,Float64}(0.37)
    nb = neighborhood(ch, ζ)
    @test CompScienceMeshes.cartesian(nb) == map(ch, ζ[1])
    @test CompScienceMeshes.parametric(nb) == ζ
    @test size(CompScienceMeshes.tangents(nb)) == (2,1)
    @test jacobian(ch, ζ[1]) > 0
end

# ------------------------------------------------------------------------------
# 2) BEAST assembly on circle with D0 basis built from the curved mesh
# ------------------------------------------------------------------------------

S = Helmholtz2D.singlelayer(wavenumber = 1.0)
X = lagrangecxd0(m)   # BEAST’s 1D C⁰ Lagrange on your curved edges

@testset "Assembly & symmetry (DoubleNumQStrat vs DoubleNumSauterQstrat)" begin
    # baseline (far/near) like you used before
    Zq = assemble(S, X, X;
                  threading = BEAST.Threading{:single},
                  quadstrat = BEAST.DoubleNumQStrat(15,14))
    @test norm(Zq - transpose(Zq)) / norm(Zq) < 1e-12

    # Sauter–Schwab path (requires your CurvilinearSimplex introspection + nodes)
    Zss = assemble(S, X, X;
                   threading = BEAST.Threading{:single},
                   quadstrat = BEAST.DoubleNumSauterQstrat(3,3,0,4,30,30))
    @test norm(Zss - transpose(Zss)) / norm(Zss) < 1e-12

    # the two should be close (not identical), tighten as needed
    @test norm(0.5*(Zq+transpose(Zq)) - 0.5*(Zss+transpose(Zss))) / norm(Zss) < 5e-3
end

@testset "Order refinement (stability)" begin
    sym(A) = 0.5 .* (A .+ transpose(A))   # complex-symmetric

    # Pick an allowed, reasonably high reference order
    qref = 20
    Zref = sym(assemble(S, X, X;
                        threading = BEAST.Threading{:single},
                        quadstrat = BEAST.DoubleNumSauterQstrat(qref,qref, qref,qref, qref,qref)))

    # Only use allowed degrees: 10, 15, 20
    for q in (10, 15, 20)
        Z = sym(assemble(S, X, X;
                         threading = BEAST.Threading{:single},
                         quadstrat = BEAST.DoubleNumSauterQstrat(q,q, q,q, q,q)))
        δ = norm(Z - Zref) / norm(Zref)
        @info "Sauter–Schwab q=$q rel. diff vs ref(q=$qref)" δ
        @test δ < 5e-3
    end
end


# ------------------------------------------------------------------------------
# 3) Circle eigenvalue checks (Rayleigh) under BEAST’s e^{+iωt} convention
# ------------------------------------------------------------------------------

@testset "Rayleigh quotient on circle (n=0…6)" begin
    κ = 3.0
    Sκ = Helmholtz2D.singlelayer(wavenumber = κ)
    sym(A) = 0.5 .* (A .+ transpose(A))

    # assemble once at moderately high SS orders
    Z = sym(assemble(Sκ, X, X;
                     threading = BEAST.Threading{:single},
                     quadstrat = BEAST.DoubleNumSauterQstrat(3,3,0,4,30,30)))

    # exact D0 mass (edge lengths)
    MD0 = Diagonal([measure(chart(m,i)) for i in cells(m)])

    # analytic λ_n for e^{+iωt}:  -½ i π r J_n(κr) H_n^{(2)}(κr)
    λ_analytic(n) = -0.5im * π * 1.0 * besselj(n, κ) * hankelh2(n, κ)

    # build D0 coefficients by edge-average of e^{i n θ} on each edge
    function d0_mode(n::Int; q=10)
        c = ComplexF64[]
        for i in cells(m)
            ch = chart(m, i)
            ξ, w = gausslegendre(q)
            s = 0.0 + 0.0im
            len = measure(ch)
            @inbounds for k in eachindex(ξ)
                ζ  = (ξ[k] + 1)/2
                x  = map(ch, ζ)
                θ  = atan(x[2], x[1]) % (2π)
                s += cis(n*θ) * jacobian(ch, ζ) * 0.5 * w[k]
            end
            push!(c, s/len)
        end
        c
    end

    for n in 0:6
        c = d0_mode(n; q=12)
        λ_RQ = (c' * Z * c) / (c' * (MD0 * c))
        λ_ex = λ_analytic(n)
        @info "n=$n  |λ_RQ-λ_ex|/|λ_ex|" rel = abs(λ_RQ-λ_ex)/abs(λ_ex)
        @test isapprox(λ_RQ, λ_ex; rtol=3e-2)
    end
end

# ------------------------------------------------------------------------------
# 4) Sauter–Schwab contact classification using your charts
# ------------------------------------------------------------------------------

@testset "Sauter–Schwab contact classification (_edgehits)" begin
    # pick two consecutive edges → share a vertex
    is = first(cells(m))
    it = is == last(cells(m)) ? first(cells(m)) : is + 1
    τ  = chart(m, is)
    σ  = chart(m, it)

    # coincident
    @test BEAST._numhits(τ, τ; tol=1e-12) == 2
    # common vertex
    @test BEAST._numhits(τ, σ; tol=1e-12) == 1
    # clearly disjoint (skip some)
    k  = is + 5
    k > last(cells(m)) && (k -= length(cells(m)))
    ρ  = chart(m, k)
    @test BEAST._numhits(τ, ρ; tol=1e-12) == 0
end

println("\nAll CurvilinearSimplex-based tests completed.\n")




edgehits_local(chs::CompScienceMeshes.CurvilinearSimplex{U,1},
               cht::CompScienceMeshes.CurvilinearSimplex{U,1};
               tol::Real=1e-12) where {U} =
begin
    s0, s1 = CompScienceMeshes.nodes(chs)
    t0, t1 = CompScienceMeshes.nodes(cht)
    d00 = norm(s0 - t0); d01 = norm(s0 - t1)
    d10 = norm(s1 - t0); d11 = norm(s1 - t1)
    if (d00 ≤ tol && d11 ≤ tol) || (d01 ≤ tol && d10 ≤ tol); 2
    elseif d00 ≤ tol || d01 ≤ tol || d10 ≤ tol || d11 ≤ tol; 1
    else; 0 end
end

@testset "Sauter–Schwab contact classification (geometry-based)" begin
    is = first(cells(m))
    it = is == last(cells(m)) ? first(cells(m)) : is + 1
    τ  = chart(m, is)
    σ  = chart(m, it)
    k  = is + 5; k > last(cells(m)) && (k -= length(cells(m)))
    ρ  = chart(m, k)

    @test edgehits_local(τ, τ) == 2
    @test edgehits_local(τ, σ) == 1
    @test edgehits_local(τ, ρ) == 0
end


const SSQ1D = BEAST.SauterSchwabQuadrature1D

@testset "Self pair ⇒ CommonEdge rule or Adjacent edges ⇒ CommonVertex rule" begin
    S  = Helmholtz2D.singlelayer(wavenumber=1.0)
    X  = lagrangecxd0(m)
    g  = BEAST.refspace(X)
    qs = BEAST.DoubleNumSauterQstrat(10,10, 10,10, 10,10)  # supported degrees

    nc = length(cells(m))
    i  = first(cells(m))
    τ  = chart(m, i)

    ip1 = mod1(i+1, nc)          # always a valid “next” edge
    σ   = chart(m, ip1)

    qd  = BEAST.quaddata(S, g, g, [τ], [σ], qs)

    # coincident → CommonEdge
    @test BEAST.quadrule(S, g, g, 1, τ, 1, τ, qd, qs) isa SSQ1D.CommonEdge
    # share a vertex → CommonVertex
    @test BEAST.quadrule(S, g, g, 1, τ, 1, σ, qd, qs) isa SSQ1D.CommonVertex

    # pick a clearly disjoint edge (avoid neighbors)
    k = mod1(i + 5, nc)
    if k == i || k == ip1 || k == mod1(i-1, nc)
        k = mod1(i + nc ÷ 2, nc)  # fallback far away
    end
    ρ = chart(m, k)

    r = BEAST.quadrule(S, g, g, 1, τ, 1, ρ, qd, qs)
    @test !(r isa SSQ1D.CommonEdge) && !(r isa SSQ1D.CommonVertex)  # disjoint → product rule
end

#=
const SSQ1D = BEAST.SauterSchwabQuadrature1D
@testset "Self pair ⇒ CommonEdge rule or Adjacent edges ⇒ CommonVertex rule" begin
    S  = Helmholtz2D.singlelayer(wavenumber=1.0)
    X  = lagrangecxd0(m)                         # your D0 basis on the curved mesh
    g  = BEAST.refspace(X)                       # ← this avoids constructor details
    qs = BEAST.DoubleNumSauterQstrat(10,10, 10,10, 10,10)

    i  = first(cells(m))
    τ  = chart(m, i)
    σ  = chart(m, i == last(cells(m)) ? first(cells(m)) : i+1)

    qd = BEAST.quaddata(S, g, g, [τ], [σ], qs)
    @test BEAST.quadrule(S, g, g, 1, τ, 1, τ, qd, qs) isa SSQ1D.CommonEdge
    @test BEAST.quadrule(S, g, g, 1, τ, 1, σ, qd, qs) isa SSQ1D.CommonVertex

    k = i + 5
    if k > length(cells(m))
        k -= length(cells(m))
    end

    ρ = chart(m, k)
    r = BEAST.quadrule(S, g, g, 1, τ, 1, ρ, qd, qs)
    @test !(r isa SSQ1D.CommonEdge) && !(r isa SSQ1D.CommonVertex)  # disjoint → product rule
end
=#