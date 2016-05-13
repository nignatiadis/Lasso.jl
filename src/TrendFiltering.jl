module TrendFiltering
using StatsBase, Isotonic, ..FusedLassoMod, ..Util
import Base: +, -, *, \, length, size, linearindexing
export TrendFilter, IsotonicTrendFilter, DifferenceMatrix,
       FallingFactorialMatrix,
       FallingFactorialModel
export EquidistantGrid, Design

include("FallingFactorial.jl")
include("TrendFilteringMatrices.jl")
# Implements the algorithm from Ramdas, A., & Tibshirani, R. J. (2014).
# Fast and flexible ADMM algorithms for trend filtering. arXiv
# Preprint arXiv:1406.2082. Retrieved from
# http://arxiv.org/abs/1406.2082


# Sum of squared differences between two vectors
function sumsqdiff(x, y)
    length(x) == length(y) || throw(DimensionMismatch())
    v = zero(Base.promote_eltype(x, y))
    @simd for i = 1:length(x)
        @inbounds v += abs2(x[i] - y[i])
    end
    v
end

# Soft threshold
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

include("TrendFilteringFit.jl")

end
