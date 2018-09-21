# simple program to test the new k-means (not ready yet)

using Test
using Clustering
using Distances
using LinearAlgebra

import Distances.pairwise!

# custom distance metric
mutable struct MySqEuclidean <: SemiMetric end

# redefinition of Distances.pairwise! for MySqEuclidean type
function pairwise!(r::AbstractMatrix, dist::MySqEuclidean,
                   a::AbstractMatrix, b::AbstractMatrix)
    mul!(r, transpose(a), b)
    sa2 = sum(abs2, a, dims=1)
    sb2 = sum(abs2, b, dims=1)
    @inbounds r .= sa2' .+ sb2 .- 2r
end

@testset "kmeans() (sparse k-means)" begin

Random.seed!(34568)

m = 5 # features
n = 1000 # samples
k = 10 # centers
sparsity = 3

x = rand(m, n)

@testset "non-weighted" begin
    r = kmeans(x, k; maxiter=50, sparsity=sparsity)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost
    @test nunsharedfeatures(r) == sparsity
end

@testset "non-weighted (float32)" begin
    r = kmeans(map(Float32, x), k; maxiter=50, sparsity=sparsity)
    @test isa(r, KmeansResult{Float32})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost
    @test nunsharedfeatures(r) == sparsity
end

@testset "weighted" begin
    w = rand(n)
    r = kmeans(x, k; maxiter=50, weights=w, sparsity=sparsity)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test nunsharedfeatures(r) == sparsity

    cw = zeros(k)
    for i = 1:n
        cw[r.assignments[i]] += w[i]
    end
    @test r.cweights ≈ cw
    @test dot(r.costs, w) ≈ r.totalcost
end

@testset "custom distance" begin
    r = kmeans(x, k; maxiter=50, init=:kmcen, distance=MySqEuclidean(), sparsity=sparsity)
    r2 = kmeans(x, k; maxiter=50, init=:kmcen, sparsity=sparsity)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost
    for fn in fieldnames(typeof(r))
        @test getfield(r, fn) == getfield(r2, fn)
    end
    @test nunsharedfeatures(r) == sparsity
end

end
