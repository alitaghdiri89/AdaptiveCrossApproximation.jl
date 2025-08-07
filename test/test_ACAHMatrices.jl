using BEAST
using HMatrices
using CompScienceMeshes
using AdaptiveCrossApproximation
using Test
using LinearAlgebra


Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
xclt = ClusterTree(spaceX.pos)

K = HMatrices.KernelMatrix(op, spaceX, spaceX)
permK = HMatrices.PermutedMatrix(K, HMatrices.loc2glob(xclt), HMatrices.loc2glob(xclt))

# rtols = [10.0^i for i in collect(-4:-1:-10)]# 1e-4 to 1e-10
rtols = [1e-4]
tst_vec = rand(dim)

fullmat = BEAST.assemble(op, spaceX, spaceX)
trueResult = fullmat * tst_vec;
aca = ACA()
hmat_Adaptive = HMatrices.assemble_hmatrix(K; comp=aca);
norm(hmat_Adaptive * tst_vec - trueResult) / norm(trueResult)
##

for rtol in rtols
    K = HMatrices.KernelMatrix(op, spaceX, spaceX)
    aca = ACA()
    hmat_Adaptive = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat_Adaptive, 1) == size(fullmat, 1)
    @test size(hmat_Adaptive, 2) == size(fullmat, 2)
    @test norm(hmat_Adaptive * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end
##

vov = HMatrices.VectorOfVectors(Float64, 3, 3);
mat = Matrix(vov.data, 3, 3)
cnt = 0
for i in 1:3
    for j in 1:3
        cnt += 1
        vov[i, j] = cnt
    end
end
##
using Plots
rel_errors = [
    4.1786664869569356e-6, 
    1.4567186187601887e-7,
    1.0935250918321565e-8,
    1.5373798468233364e-9,
    1.390990819013863e-10,
    6.848790599222273e-12,
    9.430670698663698e-13]
rtols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

plot(rtols, rel_errors)
scatter!(rtols, rel_errors)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")
png("err_vs_tol_adaptive_ext")