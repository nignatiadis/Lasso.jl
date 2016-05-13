immutable EquidistantGrid <: AbstractArray{Int, 1}
    n::Int
end

Base.length(x::EquidistantGrid) = x.n
Base.size(x::EquidistantGrid) = (x.n,)
Base.linearindexing(::Type{EquidistantGrid}) = Base.LinearFast()
Base.getindex(x::EquidistantGrid, i::Int) = i

defaultρ(x::EquidistantGrid, λ) = λ #choosing ADMM parameter ρ

typealias Design{T} Union{EquidistantGrid, Vector{T}}



immutable FallingFactorialMatrix{T, D<: Design} <: AbstractMatrix{T}
    k::Int
    x::D
end

function call{T}(::Type{FallingFactorialMatrix{T}}, k::Int, x::Design{T})
  FallingFactorialMatrix{T, Design{T}}(k,x)
end


function call{T}(::Type{FallingFactorialMatrix{T}}, k::Int, n::Int)
  FallingFactorialMatrix{T}(k, EquidistantGrid(n))
end

Base.size(mat::FallingFactorialMatrix) = (length(mat.x), length(mat.x))

function Base.LinAlg.A_mul_B!(A::FallingFactorialMatrix, y::AbstractVector)
    k = A.k
    n = length(A.x)
    for i = k:-1:0
        fac = max(i,1)
        y[(i+1):n] = cumsum(sub(y,(i+1):n)).*fac
    end
    y
end

function Base.LinAlg.A_mul_B!(out::AbstractVector,A::FallingFactorialMatrix, y::AbstractVector)
    copy!(out,y)
    A_mul_B!(A, out)
end

function *(A::FallingFactorialMatrix, y::AbstractVector)
    out = copy(y)
    A_mul_B!(A, out)
end

function *(A::FallingFactorialMatrix, B::AbstractMatrix)
    out = copy(B)
    for i=1:size(B,2)
        out[:,i] = A*vec(B[:,i])
    end
    out
end

function Base.LinAlg.A_ldiv_B!(A::FallingFactorialMatrix, y::AbstractVector)
    k = A.k
    n = length(A.x)
    @inbounds for i = 0:k
        fac = max(i,1)
        y[n] ./= fac
        @inbounds for j = n:-1:(i+2)
            y[j-1] ./= fac
            y[j] -= y[(j-1)]
        end
    end
    y
end


function Base.LinAlg.A_ldiv_B!(out::AbstractVector,A::FallingFactorialMatrix, y::AbstractVector)
    copy!(out,y)
    A_ldiv_B!(A, out)
end

function \(A::FallingFactorialMatrix, y::AbstractVector)
    out = copy(y)
    A_ldiv_B!(A, out)
end


function \(A::FallingFactorialMatrix, B::AbstractMatrix)
    out = copy(B)
    for i=1:size(B,2)
        out[:,i] = A\vec(B[:,i])
    end
    out
end

immutable FallingFactorialModel{T} <: StatisticalModel
    A:: FallingFactorialMatrix{T}
    coef:: Vector{T}
end

StatsBase.coef(t::FallingFactorialModel) = t.coef
StatsBase.predict(t::FallingFactorialModel) = t.A*coef(t)

function StatsBase.predict{T}(model::FallingFactorialModel{T}, xnew::T)
    x = model.A.x
    n = length(x)
    k = model.A.k
    ynew = zero(T)
    pol = one(T)
    for i=1:(k+1)
        ynew += coef(model)[i] * pol
        pol *= (xnew - x[i])
    end

    for j=1:(n-k-1)
        if xnew > x[j+k]
            pol = one(T)
            if k >= 1
                for l=1:k
                    pol *= (xnew-x[j+l])
                end
            end
            ynew += coef(model)[k+1+j] * pol
        end
    end
    ynew
end

function StatsBase.predict{T}(model::FallingFactorialModel{T}, xnew::Vector{T})
    ynew = similar(xnew)
    for (i,xi) in enumerate(xnew)
        ynew[i] = predict(model, xi)
    end
    ynew
end
