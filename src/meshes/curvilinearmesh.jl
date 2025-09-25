#############################
# Curvilinear 1D mesh + chart
#############################

using StaticArrays
import LinearAlgebra: norm
import Base: map

# Mesh: 1D elements embedded in R^U, with N control points per element (N = order+1)
mutable struct CurvilinearMesh{U,N,T,O} <: AbstractMesh{U,N,T}
    vertices::Vector{SVector{U,T}}   # coordinates in R^U
    faces::Vector{SVector{N,Int}}    # connectivity (local order = Gmsh order)

    function CurvilinearMesh(
        vertices::Vector{SVector{U,T}},
        faces::Vector{SVector{N,Int}},
        order::Integer
    ) where {U,N,T}
        order > 0 || throw(ArgumentError("order must be positive."))
        new{U,N,T,Int(order)}(vertices, faces)
    end
end



# Constructors from abstract containers
CurvilinearMesh(vertices::AbstractVector{<:SVector{U,T}},
                faces::AbstractVector{<:SVector{N,Int}},
                order::Integer) where {U,N,T} =
    CurvilinearMesh(Vector{SVector{U,T}}(vertices),
                    Vector{SVector{N,Int}}(faces),
                    order)

# From matrices (verts: U×nV, faces: N×nF)
function CurvilinearMesh(verts::AbstractMatrix{T},
                         faces::AbstractMatrix{<:Integer},
                         order::Integer) where {T}
    U = size(verts, 1)
    N = size(faces, 1)
    v = [SVector{U,T}(verts[:, i]) for i in 1:size(verts, 2)]
    f = [SVector{N,Int}(faces[:, i]) for i in 1:size(faces, 2)]
    CurvilinearMesh(v, f, order)
end

# Mesh interface
coordtype(::CurvilinearMesh{U,N,T,O}) where {U,N,T,O} = T
vertextype(::CurvilinearMesh{U,N,T,O}) where {U,N,T,O} = SVector{U,T}
universedimension(::CurvilinearMesh{U}) where {U} = U
dimension(::CurvilinearMesh) = 1

vertices(m::CurvilinearMesh) = m.vertices
faces(m::CurvilinearMesh)    = m.faces

numvertices(m::CurvilinearMesh) = length(m.vertices)
numcells(m::CurvilinearMesh)    = length(m.faces)
cells(m::CurvilinearMesh)       = Base.OneTo(length(m.faces))
cell(m::CurvilinearMesh, i::Int) = m.faces[i]

Base.eltype(::Type{CurvilinearMesh{U,N,T,O}}) where {U,N,T,O} = SVector{N,Int}
Base.length(m::CurvilinearMesh) = numcells(m)
Base.iterate(m::CurvilinearMesh, i::Int=1) =
    i > length(m.faces) ? nothing : (m.faces[i], i+1)

mesh_order(::CurvilinearMesh{U,N,T,O}) where {U,N,T,O} = O

########################################
# 1D Curvilinear simplex (the "chart")
########################################

"""
CurvilinearSimplex: 1D element in R^U with N control points.
C = U-1 (complementary dimension).
"""
#=
struct CurvilinearSimplex{U,D,C,N,T} <: AbstractSimplex{U,D}
    vertices::NTuple{N,SVector{U,T}}   # control points in local (element) order
    ζnodes::NTuple{N,Float64}          # reference nodes on [0,1] matching that order
end
=#

struct CurvilinearSimplex{U,D,C,N,T} <: AbstractSimplex{U,D}
    vertices::NTuple{N,SVector{U,T}}
    ζnodes::NTuple{N,T}               # ← was Float64
end

# endpoints of a 1D curvilinear element (by ζ = 0 and ζ = 1)
@inline function nodes(s::CurvilinearSimplex{U,1}) where {U}
    ζ = s.ζnodes
    Tζ = eltype(ζ)
    i0 = findfirst(==(zero(Tζ)), ζ)
    i1 = findfirst(==(one(Tζ)),  ζ)
    return (s.vertices[i0], s.vertices[i1])
end

# convenience
vertices(s::CurvilinearSimplex) = s.vertices
#refnodes(s::CurvilinearSimplex) = s.ζnodes

# value-form
coordtype(::CurvilinearSimplex{U,D,C,N,T}) where {U,D,C,N,T} = T
dimension(::CurvilinearSimplex{U,D,C,N,T})  where {U,D,C,N,T} = D
universedimension(::CurvilinearSimplex{U,D,C,N,T}) where {U,D,C,N,T} = U
vertextype(::CurvilinearSimplex{U,D,C,N,T}) where {U,D,C,N,T} = SVector{U,T}

# type-form
coordtype(::Type{CurvilinearSimplex{U,D,C,N,T}}) where {U,D,C,N,T} = T
dimension(::Type{CurvilinearSimplex{U,D,C,N,T}})  where {U,D,C,N,T} = D
universedimension(::Type{CurvilinearSimplex{U,D,C,N,T}}) where {U,D,C,N,T} = U
vertextype(::Type{CurvilinearSimplex{U,D,C,N,T}}) where {U,D,C,N,T} = SVector{U,T}



# for 1D only
#=
function nodes(s::CurvilinearSimplex{U,1}) where {U}
    (s.vertices[1], s.vertices[2])
end
=#

#@inline function nodes(s::CurvilinearSimplex{U,1}) where {U}
#    ζ = s.ζnodes
#    i0 = findfirst(==(zero(eltype(ζ))), ζ)
#    i1 = findfirst(==(one(eltype(ζ))),  ζ)
#    return (s.vertices[i0], s.vertices[i1])
#end


# Build the chart (element) from a connectivity tuple
#function chart(m::CurvilinearMesh{U,N,T,O}, conn::SVector{N,Int}) where {U,N,T,O}
#    X = ntuple(r -> m.vertices[conn[r]], N)      # control points in Gmsh local order
#    # reference abscissae on [0,1] in the SAME local order
#    #ζnodes = N == 3 ? (0.0, 1.0, 0.5) : ntuple(r -> (r-1)/(N-1), N)
#    ζnodes = N == 3 ? (0.0, 0.5, 1.0) : ntuple(r -> (r-1)/(N-1), N)
#    CurvilinearSimplex{U,1,U-1,N,T}(X, ζnodes)
#end

chart(m::CurvilinearMesh, i::Int) = chart(m, m.faces[i])

# Lagrange basis on [0,1] through ζnodes
@inline function _ℓ(ζnodes::NTuple{D,T}, r::Int, ζ::T) where {D,T<:Real}
    ζr = ζnodes[r]
    num = one(T); den = one(T)
    @inbounds for j in 1:D
        j == r && continue
        num *= (ζ - ζnodes[j])
        den *= (ζr - ζnodes[j])
    end
    num/den
end

@inline function _dℓ(ζnodes::NTuple{D,T}, r::Int, ζ::T) where {D,T<:Real}
    ζr = ζnodes[r]
    s = zero(T)
    @inbounds for k in 1:D
        k == r && continue
        num = one(T); den = (ζr - ζnodes[k])
        @inbounds for j in 1:D
            (j == r || j == k) && continue
            num *= (ζ - ζnodes[j])
            den *= (ζr - ζnodes[j])
        end
        s += num/den
    end
    s
end

#import Base: map

@inline function map(ch::CurvilinearSimplex{U,1,C,N,T}, ζ::Real) where {U,C,N,T}
    z = T(ζ)
    s = zero(SVector{U,T})
    @inbounds for r in 1:N
        s += ch.vertices[r] * _ℓ(ch.ζnodes, r, z)
    end
    s
end

@inline function dmap(ch::CurvilinearSimplex{U,1,C,N,T}, ζ::Real) where {U,C,N,T}
    z = T(ζ)
    s = zero(SVector{U,T})
    @inbounds for r in 1:N
        s += ch.vertices[r] * _dℓ(ch.ζnodes, r, z)
    end
    s
end

# Endpoints from ζ=0 and ζ=1 regardless of vertex order
#@inline function nodes(s::CompScienceMeshes.CurvilinearSimplex{U,1}) where {U}
#    ζ = s.ζnodes
#    i0 = findfirst(==(zero(eltype(ζ))), ζ)
#    i1 = findfirst(==(one(eltype(ζ))),  ζ)
#    return (s.vertices[i0], s.vertices[i1])
#end

# Gmsh-consistent ζ-nodes for a 1D line with N local nodes
gmsh_line_ζnodes(::Val{N}) where {N} =
    N == 2 ? (0.0, 1.0) :
             (0.0, 1.0, ((k)/(N-1) for k in 1:N-2)...)

function chart(m::CompScienceMeshes.CurvilinearMesh{U,N,T,O},
               conn::SVector{N,Int}) where {U,N,T,O}
    X = ntuple(r -> m.vertices[conn[r]], N)             # Gmsh local order
    ζnodes = gmsh_line_ζnodes(Val(N))                   # match that order
    CompScienceMeshes.CurvilinearSimplex{U,1,U-1,N,T}(X, ζnodes)
end



#=
@inline function _ℓ(ζnodes::NTuple{D,Float64}, r::Int, ζ) where {D}
    ζr = ζnodes[r]
    num = 1.0; den = 1.0
    @inbounds for j in 1:D
        j == r && continue
        num *= (ζ - ζnodes[j])
        den *= (ζr - ζnodes[j])
    end
    num/den
end

@inline function _dℓ(ζnodes::NTuple{D,Float64}, r::Int, ζ) where {D}
    ζr = ζnodes[r]
    s = 0.0
    @inbounds for k in 1:D
        k == r && continue
        num = 1.0; den = (ζr - ζnodes[k])
        @inbounds for j in 1:D
            (j == r || j == k) && continue
            num *= (ζ - ζnodes[j])
            den *= (ζr - ζnodes[j])
        end
        s += num/den
    end
    s
end

# physical map x(ζ), tangent ∂x/∂ζ, Jacobian J = |∂x/∂ζ|
@inline function map(ch::CurvilinearSimplex{U,1,C,N,T}, ζ::Real) where {U,C,N,T}
    s = zero(SVector{U,T})
    z = T(ζ)
    @inbounds for r in 1:N
        s += ch.vertices[r] * T(_ℓ(ch.ζnodes, r, z))
    end
    s
end

@inline function dmap(ch::CurvilinearSimplex{U,1,C,N,T}, ζ::Real) where {U,C,N,T}
    s = zero(SVector{U,T})
    z = T(ζ)
    @inbounds for r in 1:N
        s += ch.vertices[r] * T(_dℓ(ch.ζnodes, r, z))
    end
    s
end

#@inline function map(ch::CurvilinearSimplex{U,1,C,N,T}, ζ) where {U,C,N,T}
#    s = zero(SVector{U,T})
#    @inbounds for r in 1:N
#        s += ch.vertices[r] * (one(T) * _ℓ(ch.ζnodes, r, ζ))
#    end
#    s
#end
#
#@inline function dmap(ch::CurvilinearSimplex{U,1,C,N,T}, ζ) where {U,C,N,T}
#    s = zero(SVector{U,T})
#    @inbounds for r in 1:N
#        s += ch.vertices[r] * (one(T) * _dℓ(ch.ζnodes, r, ζ))
#    end
#    s
#end
=#

jacobian(ch::CurvilinearSimplex{U,1}, ζ::Real) where {U} = norm(dmap(ch, ζ))
center(ch::CurvilinearSimplex) = map(ch, 0.5)

# domain of the 1D chart (unit segment reference)
#domain(::CurvilinearSimplex{U,1}) where {U} = ReferenceSimplex{1,Float64,2}()

paramdim(::CurvilinearSimplex{U,1}) where {U} = 1
domain(::CurvilinearSimplex{U,1,C,N,T}) where {U,C,N,T} = ReferenceSimplex{1,T,2}()

# a handy geometric measure (edge length by 3-pt Gauss)
#=
function measure(ch::CurvilinearSimplex{U,1}) where {U}
    ξ = (-sqrt(3/5), 0.0, sqrt(3/5))
    w = (5/9, 8/9, 5/9)
    # map [-1,1]→[0,1]: ζ = (ξ+1)/2, dζ = 1/2
    s = zero(eltype(ch.ζnodes))
    for k in 1:3
        ζ = (ξ[k]+1)/2
        s += w[k] * jacobian(ch, ζ) * 0.5
    end
    s
end
=#

function measure(ch::CurvilinearSimplex{U,1,C,N,T}) where {U,C,N,T}
    ξ = (-sqrt(T(3)/T(5)), zero(T),  sqrt(T(3)/T(5)))
    w = (T(5)/T(9),       T(8)/T(9), T(5)/T(9))
    s = zero(T)
    @inbounds for k in 1:3
        ζ = (ξ[k] + one(T)) / T(2)
        s += w[k] * jacobian(ch, ζ) / T(2)
    end
    s
end

#paramdim(::CurvilinearSimplex{U,1}) where {U} = 1

#nodes(s::CurvilinearSimplex{U,1}) where {U} = (s.vertices[1], s.vertices[2])
refnodes(s::CurvilinearSimplex) = s.ζnodes




