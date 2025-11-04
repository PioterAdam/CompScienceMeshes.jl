using StaticArrays
using LinearAlgebra, Statistics
using Test

using CompScienceMeshes
import CompScienceMeshes: chart, measure, jacobian, refnodes, neighborhood,
                          coordtype, dimension, universedimension, vertextype,
                          quadpoints

#using BEAST
using FastGaussQuadrature
using SpecialFunctions
using Gmsh

using LinearAlgebra: norm


"""
    circle_curvilinear(radius, porder; h = 2π*radius/64)

Return a 1D-in-2D CurvilinearMesh of a circle with polynomial order `porder`
(using Gmsh high-order line elements). `h` controls target edge length.
"""
function circle_curvilinear(radius::Real, porder::Integer; h::Real = 2π*radius/64)
    @assert porder ≥ 1 "porder must be ≥ 1"
    gmsh.initialize()
    try
        # 0: errors only, 1: warnings, 2: info (default-ish), 3+: debug
        gmsh.option.setNumber("General.Verbosity", 0)     # hide “Info …” lines
        # Optional extras to make it *really* quiet:
        #gmsh.option.setNumber("General.ProgressBar", 0)   # no progress bars

        #gmsh.option.setNumber("General.Terminal", 0)      # no terminal prints
        #gmsh.option.setString("General.LogFile", "")    # ensure no log file
        
        
        gmsh.model.add("circle_p$(porder)")
        s = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
        gmsh.model.occ.synchronize()

        # sizing + high-order
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.ElementOrder", porder)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        # physicals
        gmsh.model.addPhysicalGroup(2, [s], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        curves = [t[2] for t in gmsh.model.getBoundary([(2, s)], false, false, false) if t[1] == 1]
        gmsh.model.addPhysicalGroup(1, curves, 2)
        gmsh.model.setPhysicalName(1, 2, "Boundary")

        gmsh.model.mesh.generate(2)

        # ---- nodes on boundary physical (robust to 2- or 3-value return) ----
        nb = gmsh.model.mesh.getNodesForPhysicalGroup(1, 2)
        nodeTags   = nb[1]
        nodeCoords = nb[2]
        @assert length(nodeCoords) == 3*length(nodeTags) "unexpected coords size"

        # map tag -> local index
        tag2idx = Dict{Int,Int}(Int(nodeTags[i]) => i for i in eachindex(nodeTags))

        # vertices in any order (faces will reference via tag2idx)
        verts = Vector{SVector{2,Float64}}(undef, length(nodeTags))
        @inbounds for i in eachindex(nodeTags)
            x = nodeCoords[3(i-1)+1]; y = nodeCoords[3(i-1)+2]
            verts[i] = SVector{2,Float64}(x, y)
        end

       
        # ---- faces from boundary curves, only keep the type matching porder ----
        NF = porder + 1
 
        #B/C0# keep only the 1D blocks whose arity equals NF = porder+1
        faces = SVector{NF,Int}[]
        for c in curves
            types, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, c)
            @inbounds for k in eachindex(types)
                tags  = elemTags[k]
                nodes = elemNodeTags[k]
                ne = length(tags)
                ne == 0 && continue

                nNode_block = div(length(nodes), ne)      # nodes per element in this block
                nNode_block == NF || continue

                @assert length(nodes) == ne * nNode_block # guard internal consistency
                for e in 1:ne
                    off = (e-1) * nNode_block
                    ids = ntuple(j -> tag2idx[Int(nodes[off + j])], NF)
                    push!(faces, SVector{NF,Int}(ids))
                end
            end
        end
        @assert !isempty(faces) "No boundary line elements with p=$(porder) found."


        return CurvilinearMesh(verts, faces, porder)
    finally
        gmsh.finalize()
    end
end

# --- helpers reused across testsets ------------------------------------------

quad_measure(ch, n::Int=5) = begin
    q = CompScienceMeshes.quadpoints(ch, n)
    w = q isa Tuple{AbstractVector,AbstractVector} ? q[2] : last.(q)
    sum(w)
end

# geometry sanity (dim/order, perimeter ≈ 2πR, radii at ζ=0,.5,1, neighborhood API)
function _test_geometry_sanity(m; R=1.0, p::Int)
    @test typeof(m) <: CompScienceMeshes.CurvilinearMesh
    @test CompScienceMeshes.dimension(m) == 1
    @test CompScienceMeshes.universedimension(m) == 2
    @test CompScienceMeshes.mesh_order(m) == p
    @test all(length(conn) == p+1 for conn in m.faces)

    perim = sum(measure(chart(m, cid)) for cid in cells(m))

    # perimeter tolerance: mildly tighter with order
    perim_tol(p) = max(0.002, 0.006 * 0.7^(p-1))
    @test isapprox(perim, 2π*R; rtol=perim_tol(p))

    # radii at endpoints & midpoints
    rs = Float64[]
    for cid in cells(m)
        ch = chart(m, cid)
        for ζ in (0.0, 0.5, 1.0)
            push!(rs, norm(map(ch, ζ)))
        end
    end
    @test maximum(abs.(rs .- R)) < 5e-3

    # neighborhood API spot check
    ch = chart(m, first(cells(m)))
    ζv = SVector{1,Float64}(0.37)
    nb = neighborhood(ch, ζv)
    @test CompScienceMeshes.cartesian(nb) == map(ch, ζv[1])
    @test CompScienceMeshes.parametric(nb) == ζv
    @test size(CompScienceMeshes.tangents(nb)) == (2,1)
    @test jacobian(ch, ζv[1]) > 0
end

# endpoint semantics (ζ endpoints map to the cell endpoints)
function _test_endpoint_semantics(m)
    for (cid, _) in pairs(m.faces)
        ch      = chart(m, cid)
        ζnodes  = refnodes(ch)
        iL      = argmin(ζnodes)
        iR      = argmax(ζnodes)
        @test isapprox(map(ch, ζnodes[iL]), ch.vertices[iL]; atol=1e-12, rtol=0)
        @test isapprox(map(ch, ζnodes[iR]), ch.vertices[iR]; atol=1e-12, rtol=0)

        # explicitly check ζ==0 and ζ==1 if present in the node set
        ζscalar(z) = z isa StaticArrays.SVector ? z[1] : z
        i0 = findfirst(z -> isapprox(ζscalar(z), 0.0; atol=1e-14, rtol=0), ζnodes)
        i1 = findfirst(z -> isapprox(ζscalar(z), 1.0; atol=1e-14, rtol=0), ζnodes)
        if i0 !== nothing && i1 !== nothing
            x0 = map(ch, ζnodes[i0]); g0 = ch.vertices[i0]
            x1 = map(ch, ζnodes[i1]); g1 = ch.vertices[i1]
            @test isapprox(x0, g0; atol=1e-12, rtol=0)
            @test isapprox(x1, g1; atol=1e-12, rtol=0)
        end
    end
end

# adjacency C0 at shared vertices (position + Jacobian agree; tangents colinear)
function _test_adjacency_c0(m)
    endpoint_gids(m, faceidx) = begin
        conn = m.faces[faceidx]
        ch   = chart(m, faceidx)
        ζ    = refnodes(ch)
        iL   = argmin(ζ); iR = argmax(ζ)
        (conn[iL], conn[iR], iL, iR, ch, ζ)
    end

    E = length(cells(m))
    for k in 1:E
        gL, gR, iL1, iR1, ch1, ζ1 = endpoint_gids(m, k)
        # find neighbor sharing gR
        nbrs = Int[]
        for j in 1:E
            j == k && continue
            gL2, gR2, .. = endpoint_gids(m, j)
            if gR == gL2 || gR == gR2
                push!(nbrs, j)
            end
        end
        @test length(nbrs) == 1
        j = nbrs[1]

        gL2, gR2, iL2, iR2, ch2, ζ2 = endpoint_gids(m, j)
        use_left = (gR == gL2)
        ζR1 = ζ1[iR1]
        ζL2 = use_left ? ζ2[iL2] : ζ2[iR2]

        # positions match at the shared vertex
        @test isapprox(map(ch1, ζR1), map(ch2, ζL2); atol=1e-12, rtol=0)

        # Jacobian scalars agree
        @test isapprox(jacobian(ch1, ζR1), jacobian(ch2, ζL2); rtol=1e-10, atol=1e-14)

        # tangents are colinear (use one-sided FD to avoid normalizing zeros)
        ϵ  = eps(eltype(ζ1))^(1/3)
        ζa1, ζb1 = ζR1 - 2ϵ, ζR1 - ϵ
        t1 = normalize(map(ch1, ζb1) - map(ch1, ζa1))
        if use_left
            ζa2, ζb2 = ζL2 + ϵ, ζL2 + 2ϵ
        else
            ζa2, ζb2 = ζL2 - 2ϵ, ζL2 - ϵ
        end
        t2 = normalize(map(ch2, ζb2) - map(ch2, ζa2))
        @test isapprox(abs(dot(t1, t2)), 1.0; atol=1e-6)
    end
end

# global orientation: sum of internal turning angles ≈ ±2π
function _test_turning_angle_2pi(m)
    tangent_at(ch, ζ) = begin
        nb = neighborhood(ch, SVector(ζ))
        t  = CompScienceMeshes.tangents(nb)[:,1]
        t ./ hypot(t[1], t[2])
    end
    signed_angle(a, b) = atan(a[1]*b[2] - a[2]*b[1],  # cross_z
                              a[1]*b[1] + a[2]*b[2])  # dot

    # Kahan-compensated sum over edges (ζ:0→1 on each chart)
    angsum = 0.0; c = 0.0
    for cid in cells(m)
        ch = chart(m, cid)
        tL = tangent_at(ch, 0.0)
        tR = tangent_at(ch, 1.0)
        α  = signed_angle(tL, tR)
        y  = α - c
        t  = angsum + y
        c  = (t - angsum) - y
        angsum = t
    end
    @test isapprox(abs(angsum), 2π; rtol=7e-4, atol=1e-6)
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

# ------------------------------------------------------------
# Param sweep over mesh resolution (N) and polynomial order (p)
# ------------------------------------------------------------
const R1 = 1.0
const Ns1 = (64, 128, 256, 512)
const ps1 = (2, 3, 4, 5)

@testset verbose=true "circle_curvilinear — sweep over N and p" begin
    for p in ps1, N in Ns1
        h = 2π*R1/N
        m = circle_curvilinear(R1, p; h=h)

        @testset "p=$p, N=$N (h=$(round(h, digits=4)))" begin
            _test_geometry_sanity(m; R=R1, p=p)
            _test_endpoint_semantics(m)
            _test_adjacency_c0(m)
            _test_turning_angle_2pi(m)
        end
    end
end

@testset verbose=true "Quarter-circle arcs: continuity at π/2 + analytic tan/normal (sweep p,N)" begin
    R  = 1.0
    Ns = (64, 128, 256)           # enlarge if you want
    ps = (2, 3, 4, 5)        # p>=2 only (curvilinear)

    # helper: principal angle in [0, 2π)
    θ(x) = begin
        ang = atan(x[2], x[1])
        ang < 0 ? ang + 2π : ang
    end

    # analytic tangent/normal on the circle at point x
    analytic_dirs(x) = begin
        th = θ(x)
        τ  = SVector(-sin(th), cos(th))  # unit tangent (CCW)
        n  = SVector( cos(th), sin(th))  # outward unit normal
        τ, n
    end

    # numerical unit tangent at ζ using neighborhood() API
    num_tangent(ch, ζ) = begin
        nb = neighborhood(ch, SVector(ζ))
        t  = CompScienceMeshes.tangents(nb)[:, 1]
        t ./ hypot(t[1], t[2])
    end

    # dot clamp to avoid tiny overshoots from FP noise
    clampdot(a,b) = begin
        d = dot(a,b)
        d >  1 ? 1.0 : (d < -1 ? -1.0 : d)
    end

    # sample points per face (avoid exactly 0/1 to sidestep one-sided issues)
    sample_ζ = (0.1, 0.3, 0.5, 0.7, 0.9)

    for N in Ns, p in ps
        h = 2π*R/N
        m = circle_curvilinear(R, p; h=h)

        @testset "p=$p, N=$N (h=$(round(h; digits=4)))" begin
            # --- classify faces into two quarter arcs by midpoint angle -------
            face_mid_angle(cid) = begin
                ch = chart(m, cid)
                θ(map(ch, 0.5))
            end
            arc1 = Int[]  # [0, π/2]
            arc2 = Int[]  # [π/2, π]
            for cid in cells(m)
                ang = face_mid_angle(cid)
                if 0.0 - 1e-12 ≤ ang ≤ π/2 + 1e-12
                    push!(arc1, cid)
                elseif π/2 - 1e-12 ≤ ang ≤ π + 1e-12
                    push!(arc2, cid)
                end
            end
            @test !isempty(arc1) && !isempty(arc2)

            # --- analytic checks along each arc --------------------------------
            # tolerance can be a touch looser on very coarse meshes
            tan_atol = max(1e-6, 1e-3*h)  # unitless
            ortho_atol = 1e-10

            for cid in vcat(arc1, arc2)
                ch = chart(m, cid)
                for ζ in sample_ζ
                    x = map(ch, ζ)
                    τa, na = analytic_dirs(x)
                    τn = num_tangent(ch, ζ)
                    na_num = SVector(-τn[2], τn[1])  # rotate τn by +90°

                    @test isapprox(abs(clampdot(τn, τa)), 1.0; atol=tan_atol)
                    @test isapprox(abs(clampdot(na_num, na)), 1.0; atol=tan_atol)
                    @test isapprox(dot(τn, na_num), 0.0; atol=ortho_atol)
                end
            end

            # --- continuity at θ = π/2 ----------------------------------------
            # choose the face from arc1 whose *right* endpoint is closest to π/2
            # and the face from arc2 whose *left*  endpoint is closest to π/2
            endpoint_with_angle(cid, right::Bool) = begin
                ch = chart(m, cid)
                ζ  = right ? 1.0 : 0.0
                x  = map(ch, ζ)
                θ(x), x, num_tangent(ch, ζ)
            end
            target = π/2

            best1 = nothing; d1 = Inf
            for cid in arc1
                th, x, t = endpoint_with_angle(cid, true)
                if abs(th - target) < d1
                    best1 = (cid=cid, θ=th, x=x, t=t); d1 = abs(th - target)
                end
            end
            best2 = nothing; d2 = Inf
            for cid in arc2
                th, x, t = endpoint_with_angle(cid, false)
                if abs(th - target) < d2
                    best2 = (cid=cid, θ=th, x=x, t=t); d2 = abs(th - target)
                end
            end
            @test best1 !== nothing && best2 !== nothing

            # Positions coincide at the junction
            @test isapprox(best1.x, best2.x; atol=1e-12, rtol=0)

            # Tangents colinear & opposite at the junction
            @test isapprox(abs(clampdot(best1.t, best2.t)), 1.0; atol=tan_atol)

            # Both match the analytic tangent at θ=π/2 → (-1, 0)
            τa_mid = SVector(-1.0, 0.0)
            @test isapprox(abs(clampdot(best1.t, τa_mid)), 1.0; atol=tan_atol)
            @test isapprox(abs(clampdot(best2.t, τa_mid)), 1.0; atol=tan_atol)
        end
    end
end
