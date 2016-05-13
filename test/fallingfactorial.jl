using Lasso, Base.Test

srand(1)
n = 10000
x =  map(Float64, bitrand(n))
#x = rand(n)
println("Falling Factorial")
for k=0:2
  println("* k=$k")
  mat = FallingFactorialMatrix{Float64}(k,n)

  println("** invertibility test")
  @test_approx_eq_eps norm( mat*(mat\x) - x ) 0 1e-5
  @test_approx_eq_eps norm( mat\(mat*x) - x ) 0 1e-5

  out1 = Array(Float64,n-k-1)
  out2 = Array(Float64,n)

  println("** Comparison to DifferenceMatrix")

  A1 = DifferenceMatrix{Float64}(k, n)
  A2 = FallingFactorialMatrix{Float64}(k, n)

  A_mul_B!(out1, A1, x)
  A_ldiv_B!(out2, A2, x)

  @test_approx_eq norm(out1./factorial(k)-out2[(k+2):n]) .0

  println("** Statistical Model interface")
  outp = map(Float64, [1,2,2,3,3,3,2])
  A = FallingFactorialMatrix{Float64}(k,7)
  coef = A\outp
  t = FallingFactorialModel{Float64}(A,coef)
  @test_approx_eq predict(t) outp
  @test_approx_eq predict(t, [1.0;2.0;3.0;4.0;5.0;6.0;7.0]) outp
  @test 1.0 <= predict(t, 1.5) <= 2.0

end
