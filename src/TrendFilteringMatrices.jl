immutable EquidistantGrid
    n::Int
end

Base.length(x::EquidistantGrid) = x.n

typealias Design{T} Union{EquidistantGrid, Vector{T}}

immutable DifferenceMatrix{T, D<:Design} <: AbstractMatrix{T}
    k::Int
    x::D
    b::Vector{T}                  # Coefficients for A_mul_B!
    si::Vector{T}                 # State for A_mul_B!/At_mul_B!

    function DifferenceMatrix(k, x, b, si)
        n = length(x)
        n >= 2*k+2 || throw(ArgumentError("signal must have length >= 2*order+2"))
        new(k, x, b, si)
    end
end

function call{T}(::Type{DifferenceMatrix{T}}, k::Int, x::Design{T})
  n = length(x)
  n >= 2*k+2 || throw(ArgumentError("signal must have length >= 2*order+2"))
  b = T[ifelse(isodd(i), -1, 1)*binomial(k+1, i) for i = 0:k+1]
  DifferenceMatrix{T, Design{T}}(k, x, b, zeros(T, k+1))
end

function call{T}(::Type{DifferenceMatrix{T}}, k::Int, n::Int)
  DifferenceMatrix{T}(k, EquidistantGrid(n))
end

function DifferenceMatrix{T}(k::Int, x::Design{T})
    DifferenceMatrix{T}(k, x)
end

function Base.size(K::DifferenceMatrix)
    n = length(K.x)
   (n-K.k-1, n)
end

# Multiply by difference matrix by filtering
function Base.LinAlg.A_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
    length(x) == size(K, 2) || throw(DimensionMismatch())
    length(out) == size(K, 1) || throw(DimensionMismatch())
    b = K.b
    si = fill!(K.si, 0)
    silen = length(b)-1
    @inbounds for i = 1:length(x)
        xi = x[i]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        if i > silen
            out[i-silen] = α*val
        end
    end
    out
end
*(K::DifferenceMatrix, x::AbstractVector) = A_mul_B!(similar(x, size(K, 1)), K, x)

function Base.LinAlg.At_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
    length(x) == size(K, 1) || throw(DimensionMismatch())
    length(out) == size(K, 2) || throw(DimensionMismatch())
    b = K.b
    si = fill!(K.si, 0)
    silen = length(b)-1
    isodd(silen) && (α = -α)
    n = length(x)
    @inbounds for i = 1:n
        xi = x[i]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        out[i] = α*val
    end
    @inbounds for i = 1:length(si)
        out[n+i] = α*si[i]
    end
    out
end
Base.LinAlg.Ac_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1) = At_mul_B!(out, K, x)
Base.LinAlg.At_mul_B(K::DifferenceMatrix, x::AbstractVector) = At_mul_B!(similar(x, size(K, 2)), K, x)
Base.LinAlg.Ac_mul_B(K::DifferenceMatrix, x::AbstractVector) = At_mul_B!(similar(x, size(K, 2)), K, x)

# Product with self, efficiently
function Base.LinAlg.At_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix)
    K === K2 || error("matrix multiplication only supported with same difference matrix")
    computeDtD(K.b, length(K.x))
end

Base.LinAlg.Ac_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix) = At_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix)

function computeDtD(c, n)
    k = length(c) - 2
    sgn = iseven(k)
    cc = zeros(eltype(c), 2*length(c)-1)
    for i = 1:length(c)
        cc[i] = sgn ? -c[i] : c[i]
    end
    filt!(cc, c, [one(eltype(c))], cc)
    sides = zeros(eltype(c), 2*length(c)-2, length(c)-1)
    for j = 1:length(c)-1
        for i = 1:j
            sides[i, j] = sgn ? -c[i] : c[i]
        end
    end
    filt!(sides, c, [one(eltype(c))], sides)
    colptr = Array(Int, n+1)
    rowval = Array(Int, (k+2)*(n-k-1)+(k+1)*n)
    nzval = Array(Float64, (k+2)*(n-k-1)+(k+1)*n)
    idx = 1
    for i = 1:k+1
        colptr[i] = idx
        for j = 1:k+i+1
            rowval[idx+j-1] = j
            nzval[idx+j-1] = sides[k+2+i-j, i]
        end
        idx += k+i+1
    end
    for i = k+2:n-(k+1)
        colptr[i] = idx
        for j = 1:length(cc)
            rowval[idx+j-1] = i-k+j-2
            nzval[idx+j-1] = cc[j]
        end
        idx += length(cc)
    end
    for i = k+1:-1:1
        colptr[n-i+1] = idx
        for j = 1:i+k+1
            rowval[idx+j-1] = n-k-1-i+j
            nzval[idx+j-1] = sides[j, i]
        end
        idx += i+k+1
    end
    colptr[end] = idx
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end
