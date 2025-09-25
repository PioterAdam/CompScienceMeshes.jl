# Neighborhood support for 1D curvilinear simplices

import Base: getproperty
import CompScienceMeshes: neighborhood, coordtype

# nb.cart and nb.bary convenience accessors
function getproperty(nb::NeighborhoodLazy, s::Symbol)
    s === :cart && return cartesian(nb)
    s === :bary && return parametric(nb)
    return getfield(nb, s)
end

# Neighborhood constructors for any 1D simplex
#neighborhood(ch::AbstractSimplex{U,1}, u::SVector{1,<:Real}) where {U} = NeighborhoodLazy(ch, u)
#neighborhood(ch::AbstractSimplex{U,1}, u::Real)              where {U} = NeighborhoodLazy(ch, SVector(u))
# barycentric ↔ cartesian on the chart
#barytocart(ch::AbstractSimplex{U,1}, u::SVector{1,<:Real}) where {U} = map(ch, u[1])

# Curvilinear only (no overlap with straight Simplex)
neighborhood(ch::CurvilinearSimplex{U,1}, u::SVector{1,<:Real}) where {U} =
    NeighborhoodLazy(ch, u)

neighborhood(ch::CurvilinearSimplex{U,1}, u::Real) where {U} = begin
    T = coordtype(ch)
    NeighborhoodLazy(ch, SVector{1,T}(T(u)))
end

barytocart(ch::CurvilinearSimplex{U,1}, u::SVector{1,<:Real})   where {U} = map(ch, u[1])

cartesian(ch::AbstractSimplex{U,1}, u::SVector{1,<:Real})  where {U} = barytocart(ch, u)
parametric(ch::AbstractSimplex{U,1}, u::SVector{1,<:Real}) where {U} = u

# Tangent column (U×1)
function tangents(ch::AbstractSimplex{U,1}, u::SVector{1,<:Real}) where {U}
    t = dmap(ch, u[1])
    SMatrix{U,1,eltype(t)}(t)
end

# Optional: unit normal for planar curves
function normal(ch::AbstractSimplex{2,1}, u::SVector{1,<:Real})
    t  = dmap(ch, u[1])
    nt = SVector(-t[2], t[1])
    nt / norm(nt)
end

# Jacobian for neighborhood
jacobian(nb::NeighborhoodLazy{<:AbstractSimplex{U,1},<:SVector{1,<:Real}}) where {U} =
    jacobian(nb.chart, nb.params[1])
