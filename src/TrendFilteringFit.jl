
abstract GeneralizedTrendFiltering
type TrendFilter{T,S} <: GeneralizedTrendFiltering
    Dkp1::DifferenceMatrix{T}                # D(k+1)
    Dk::DifferenceMatrix{T}                  # D(k)
    DktDk::SparseMatrixCSC{T,Int}            # Dk'Dk
    β::Vector{T}                             # Output coefficients and temporary storage for ρD(k+1)'α + u
    u::Vector{T}                             # ADMM u
    Dkβ::Vector{T}                           # Temporary storage for D(k)*β
    Dkp1β::Vector{T}                         # Temporary storage for D(k+1)*β (aliases Dkβ)
    flsa::FusedLasso{T,S}                    # Fused lasso model
    niter::Int                               # Number of ADMM iterations
end

function StatsBase.fit{T}(::Type{TrendFilter}, y::AbstractVector{T}, order, λ; dofit::Bool=true, args...)
    order >= 1 || throw(ArgumentError("order must be >= 1"))
    Dkp1 = DifferenceMatrix{T}(order, length(y))
    Dk = DifferenceMatrix{T}(order-1, length(y))
    β = zeros(T, length(y))
    u = zeros(T, size(Dk, 1))
    Dkp1β = zeros(T, size(Dkp1, 1))
    Dkβ = pointer_to_array(pointer(Dkp1β), size(Dk, 1))
    tf = TrendFilter(Dkp1, Dk, Dk'Dk, β, u, Dkβ, Dkp1β, fit(FusedLasso, Dkβ, λ; dofit=false), -1)
    dofit && fit!(tf, y, λ; args...)
    return tf
end

function StatsBase.fit!{T}(tf::TrendFilter{T}, y::AbstractVector{T}, λ::Real; niter=100000, tol=1e-6, ρ=λ)
    @extractfields tf Dkp1 Dk DktDk β u Dkβ Dkp1β flsa
    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    # Reuse this memory
    ρDtαu = β
    αpu = Dkβ

    fact = cholfact(speye(size(Dk, 2)) + ρ*DktDk)
    α = coef(flsa)
    fill!(α, 0)

    oldobj = obj = Inf
    local iter
    for iter = 1:niter
        # Eq. 11 (update β)
        broadcast!(+, αpu, α, u)
        At_mul_B!(ρDtαu, Dk, αpu, ρ)
        β = fact\broadcast!(+, ρDtαu, ρDtαu, y)

        # Check for convergence
        A_mul_B!(Dkp1β, Dkp1, β)
        oldobj = obj
        obj = sumsqdiff(y, β)/2 + λ*sumabs(Dkp1β)
        abs(oldobj - obj) < abs(obj * tol) && break

        # Eq. 12 (update α)
        A_mul_B!(Dkβ, Dk, β)
        broadcast!(-, u, Dkβ, u)
        fit!(flsa, u, λ/ρ)

        # Eq. 13 (update u; u actually contains Dβ - u)
        broadcast!(-, u, α, u)
    end
    if abs(oldobj - obj) > abs(obj * tol)
        error("ADMM did not converge in $niter iterations")
    end

    # Save coefficients
    tf.β = β
    tf.niter = iter
    tf
end

StatsBase.coef(tf::TrendFilter) = tf.β


  # Also implement IsotonicTrendFilter

  type IsotonicTrendFilter{T,S} <: GeneralizedTrendFiltering
      Dkp1::DifferenceMatrix{T}                # D(k+1)
      Dk::DifferenceMatrix{T}                  # D(k)
      DktDk::SparseMatrixCSC{T,Int}            # Dk'Dk
      β::Vector{T}                             # Output coefficients and temporary storage for ρD(k+1)'α + u
      γ::Vector{T}                             # ADMM primal γ
      u::Vector{T}                             # ADMM dual u
      v::Vector{T}                             # ADMM dual v
      Dkβ::Vector{T}                           # Temporary storage for D(k)*β
      Dkp1β::Vector{T}                         # Temporary storage for D(k+1)*β (aliases Dkβ)
      flsa::FusedLasso{T,S}                    # Fused lasso model
      niter::Int                               # Number of ADMM iterations
  end

function StatsBase.fit{T}(::Type{IsotonicTrendFilter}, y::AbstractVector{T}, order, λ; dofit::Bool=true, args...)
    order >= 1 || throw(ArgumentError("order must be >= 1"))
    Dkp1 = DifferenceMatrix{T}(order, length(y))
    Dk = DifferenceMatrix{T}(order-1, length(y))
    β = zeros(T, length(y))
    γ = zeros(T, length(y))
    u = zeros(T, size(Dk, 1))
    v = zeros(T, length(y))
    Dkp1β = zeros(T, size(Dkp1, 1))
    Dkβ = pointer_to_array(pointer(Dkp1β), size(Dk, 1))
    tf = IsotonicTrendFilter(Dkp1, Dk, Dk'Dk, β, γ, u, v, Dkβ, Dkp1β, fit(FusedLasso, Dkβ, λ; dofit=false), -1)
    dofit && fit!(tf, y, λ; args...)
    return tf
end

function StatsBase.fit!{T}(tf::IsotonicTrendFilter{T}, y::AbstractVector{T}, λ::Real; niter=100000, tol=1e-6, ρ1=λ,ρ2=λ)
    @extractfields tf Dkp1 Dk DktDk β γ u v Dkβ Dkp1β flsa
    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    # Reuse this memory
    ρDtαu = β
    αpu = Dkβ

    fact = cholfact((1+ρ2)*speye(size(Dk, 2)) + ρ1*DktDk)
    α = coef(flsa)
    fill!(α, 0)

    oldobj = obj = Inf
    local iter
    for iter = 1:niter
        # Eq. 11 (update β)
        broadcast!(+, αpu, α, u)
        At_mul_B!(ρDtαu, Dk, αpu, ρ1) #ρ1*Dk^T*apu

        β = fact\broadcast!(+, ρDtαu, ρDtαu, y, ρ2*γ,  ρ2*v)

        # Check for convergence
        A_mul_B!(Dkp1β, Dkp1, β)
        oldobj = obj
        obj = sumsqdiff(y, β)/2 + λ*sumabs(Dkp1β)
        abs(oldobj - obj) < abs(obj * tol) && break

        # update α
        A_mul_B!(Dkβ, Dk, β)
        broadcast!(-, u, Dkβ, u)
        fit!(flsa, u, λ/ρ1) #typo in paper ρ -> ρ1

        # update u; u actually contains Dkβ - u
        broadcast!(-, u, α, u)

        # update γ
        broadcast!(-, γ, β, v)
        isotonic_regression!(γ) #

        #update v
        broadcast!(+, v, v, γ, - β)

    end
    if abs(oldobj - obj) > abs(obj * tol)
        error("ADMM did not converge in $niter iterations")
    end

    # Save coefficients
    tf.β = β
    tf.niter = iter
    tf
end

StatsBase.coef(tf::IsotonicTrendFilter) = tf.β
