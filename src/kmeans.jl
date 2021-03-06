# K-means algorithm

#### Interface

mutable struct KmeansResult{T<:AbstractFloat} <: ClusteringResult
    centers::Matrix{T}         # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{T}           # costs of the resultant assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    cweights::Vector{Float64}  # cluster weights (k)
    totalcost::Float64         # total cost (i.e. objective) (k)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
    unshared::Vector{Bool}     # whether each feature is unshared or shared across centers
end

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

function kmeans!(X::Matrix{T}, centers::Matrix{T};
                 weights=nothing,
                 maxiter::Integer=_kmeans_default_maxiter,
                 tol::Real=_kmeans_default_tol,
                 display::Symbol=_kmeans_default_display,
                 distance::SemiMetric=SqEuclidean(),
                 sparsity::Integer=size(X, 1)) where T<:AbstractFloat

    m, n = size(X)
    m2, k = size(centers)
    m == m2 || throw(DimensionMismatch("Inconsistent array dimensions."))
    (2 <= k < n) || error("k must have 2 <= k < n.")
    (1 ≤ sparsity ≤ m) || error("sparsity must be 1 ≤ sparsity ≤ m.")

    assignments = zeros(Int, n)
    costs = zeros(T, n)
    counts = Vector{Int}(undef, k)
    cweights = Vector{Float64}(undef, k)
    if weights == nothing
        globalcenters = Statistics.mean(X, dims=2)[:]
    else
        globalcenters = StatsBase.mean(X, StatsBase.weights(weights), 2)[:]
    end
    spcriterion = Vector{T}(undef, m)
    spsortidx = collect(1:m)
    unshared = fill(true, m) # whether each feature is unshared or shared

    _kmeans!(X, conv_weights(T, n, weights), centers,
             assignments, costs, counts, cweights,
             round(Int, maxiter), tol, display_level(display), distance, 
             Int(sparsity), globalcenters, spcriterion, spsortidx, unshared)
end

function kmeans(X::Matrix{T}, k::Int;
                weights=nothing,
                init=_kmeans_default_init,
                maxiter::Integer=_kmeans_default_maxiter,
                tol::Real=_kmeans_default_tol,
                display::Symbol=_kmeans_default_display,
                distance::SemiMetric=SqEuclidean(),
                sparsity::Integer=size(X, 1)) where T<:AbstractFloat

    m, n = size(X)
    (2 <= k < n) || error("k must have 2 <= k < n.")
    (1 ≤ sparsity ≤ m) || error("sparsity must be 1 ≤ sparsity ≤ m.")
    iseeds = initseeds(init, X, k)
    centers = copyseeds(X, iseeds)
    kmeans!(X, centers;
            weights=weights,
            maxiter=maxiter,
            tol=tol,
            display=display,
            distance=distance,
            sparsity=sparsity)
end

#### Core implementation

# core k-means skeleton
function _kmeans!(
    x::Matrix{T},                   # in: sample matrix (d x n)
    w::Union{Nothing, Vector{T}},      # in: sample weights (n)
    centers::Matrix{T},             # in/out: matrix of centers (d x k)
    assignments::Vector{Int},       # out: vector of assignments (n)
    costs::Vector{T},               # out: costs of the resultant assignments (n)
    counts::Vector{Int},            # out: the number of samples assigned to each cluster (k)
    cweights::Vector{Float64},      # out: the weights of each cluster
    maxiter::Int,                   # in: maximum number of iterations
    tol::Real,                      # in: tolerance of change at convergence
    displevel::Int,                 # in: the level of display
    distance::SemiMetric,           # in: function to calculate the distance with
    sparsity::Int,                  # in: number of features to keep unshared
    globalcenters::Vector{T},       # in: feature global centers (d)
    spcriterion::Vector{T},         # out: criterion for feature selection (d)
    spsortidx::Vector{Int},         # out: sort index for feature selection (d)
    unshared::Vector{Bool}          # out: whether each feature is unshared or shared (d)
    ) where T<:AbstractFloat

    # initialize

    d, k = size(centers) # d = number of features, k = number of centers
    to_update = Vector{Bool}(undef, k) # indicators of whether a center needs to be updated
    unused = Int[] # indices of centers with no members
    num_affected::Int = k # number of centers, to which the distances need to be recomputed

    dmat = pairwise(distance, centers, x)
    dmat = convert(Array{T}, dmat) #Can be removed if one day Distance.result_type(SqEuclidean(), T, T) == T
    update_assignments!(dmat, true, assignments, costs, counts, to_update, unused)
    objv = w === nothing ? sum(costs) : dot(w, costs)

    # main loop
    t = 0
    converged = false
    if displevel >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
        @printf("%7d %18.6e\n", t, objv)
    end

    while !converged && t < maxiter
        t = t + 1

        # update (affected) centers

        update_centers!(x, w, assignments, to_update, centers, cweights, 
            sparsity, globalcenters, spcriterion, spsortidx, unshared)
            
        isempty(unused) || repick_unused_centers(x, costs, centers, unused)

        # update pairwise distance matrix

        isempty(unused) || (to_update[unused] .= true)

        if t == 1 || num_affected > 0.75 * k
            pairwise!(dmat, distance, centers, x)
        else
            # if only a small subset is affected, only compute for that subset
            pairwise!(dmat[to_update, :], distance, centers[:, to_update], x)
        end

        # update assignments

        update_assignments!(dmat, false, assignments, costs, counts, to_update, unused)
        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence

        prev_objv = objv
        objv = w === nothing ? sum(costs) : dot(w, costs)
        objv_change = objv - prev_objv

        if t > 1 && objv_change > tol && displevel >= 2
            @warn("The objective value changes towards an opposite direction")
        end

        if abs(objv_change) < tol
            converged = true
        end

        # display iteration information (if asked)

        if displevel >= 2
            @printf("%7d %18.6e %18.6e | %8d\n", t, objv, objv_change, num_affected)
        end
    end

    if displevel >= 1
        if converged
            println("K-means converged with $t iterations (objv = $objv)")
        else
            println("K-means terminated without convergence after $t iterations (objv = $objv)")
        end
    end

    return KmeansResult(centers, assignments, costs, counts, cweights,
                        Float64(objv), t, converged, unshared)
end


#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!(
    dmat::Matrix{T},            # in:  distance matrix (k x n)
    is_init::Bool,              # in:  whether it is the initial run
    assignments::Vector{Int},   # out: assignment vector (n)
    costs::Vector{T},           # out: costs of the resultant assignment (n)
    counts::Vector{Int},        # out: number of samples assigned to each cluster (k)
    to_update::Vector{Bool},    # out: whether a center needs update (k)
    unused::Vector{Int}         # out: the list of centers get no samples assigned to it
    ) where T<:AbstractFloat        

    k::Int, n::Int = size(dmat)

    # re-initialize the counting vector
    fill!(counts, 0)

    if is_init
        fill!(to_update, true)
    else
        fill!(to_update, false)
        isempty(unused) || empty!(unused)
    end

    # process each sample
    @inbounds for j = 1 : n

        # find the closest cluster to the j-th sample
        c, a = findmin(view(dmat, :, j))

        # set/update the assignment
        if is_init
            assignments[j] = a
        else  # update
            pa = assignments[j]
            if pa != a
                # if assignment changes,
                # both old and new centers need to be updated
                assignments[j] = a
                to_update[a] = true
                to_update[pa] = true
            end
        end

        # set costs and counts accordingly
        costs[j] = c
        counts[a] += 1
    end

    # look for centers that have no associated samples

    for i = 1 : k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are not weighted)
#
function update_centers!(
    x::Matrix{T},                   # in: sample matrix (d x n)
    w::Nothing,                     # in: sample weights
    assignments::Vector{Int},       # in: assignments (n)
    to_update::Vector{Bool},        # in: whether a center needs update (k)
    centers::Matrix{T},             # out: updated centers (d x k)
    cweights::Vector,               # out: updated cluster weights (k)
    sparsity::Int,                  # in: number of features to keep unshared     
    globalcenters::Vector{T},       # in: feature global centers (d)
    spcriterion::Vector{T},         # out: criterion for feature selection (d)
    spsortidx::Vector{Int},         # out: sort index for feature selection (d)
    unshared::Vector{Bool}          # out: whether each feature is unshared or not
    ) where T<:AbstractFloat

    d::Int = size(x, 1)
    n::Int = size(x, 2)
    k::Int = size(centers, 2)

    # initialize center weights
    for i = 1 : k
        if to_update[i]
            cweights[i] = 0.
        end
    end

    # accumulate columns
    @inbounds for j = 1 : n
        cj = assignments[j]
        1 <= cj <= k || error("assignment out of boundary.")
        if to_update[cj]
            if cweights[cj] > 0
                for i = 1:d
                    centers[i, cj] += x[i, j]
                end
            else
                for i = 1:d
                    centers[i, cj] = x[i, j]
                end
            end
            cweights[cj] += 1
        end
    end

    # sum ==> mean
    for j = 1:k
        if to_update[j]
            @inbounds cj::T = 1 / cweights[j]
            vj = view(centers, :, j)
            for i = 1:d
                @inbounds vj[i] *= cj
            end
        end
    end

    # sparse feature selection
    if sparsity < d
        # compute the sparsity criterion
        spcriterion .= - n .* globalcenters .* globalcenters
        for j in 1:k, i in 1:d # j center, i feature
            spcriterion[i] += cweights[j] * centers[i, j] * centers[i, j]
        end
        # replace shared feature centers by global feature centers
        fill!(unshared, true)
        sortperm!(spsortidx, spcriterion, initialized=true)
        for i = 1:(d - sparsity) # shared features
            unshared[spsortidx[i]] = false
            centers[spsortidx[i], :] .= globalcenters[spsortidx[i]]
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are weighted)
#
function update_centers!(
    x::Matrix{T},                   # in: sample matrix (d x n)
    weights::Vector{T},             # in: sample weights (n)
    assignments::Vector{Int},       # in: assignments (n)
    to_update::Vector{Bool},        # in: whether a center needs update (k)
    centers::Matrix{T},             # out: updated centers (d x k)
    cweights::Vector,               # out: updated cluster weights (k)
    sparsity::Int,                  # in: number of features to keep unshared     
    globalcenters::Vector{T},       # in: feature global centers (d)
    spcriterion::Vector{T},         # out: criterion for feature selection (d)
    spsortidx::Vector{Int},         # out: sort index for feature selection (d)
    unshared::Vector{Bool}          # out: indicate each feature is unshared or shared
    ) where T<:AbstractFloat

    d::Int = size(x, 1)
    n::Int = size(x, 2)
    k::Int = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0.0

    # accumulate columns
    # accumulate_cols_u!(centers, cweights, x, assignments, weights, to_update)
    for j = 1 : n
        @inbounds wj = weights[j]

        if wj > 0
            @inbounds cj = assignments[j]
            1 <= cj <= k || error("assignment out of boundary.")

            if to_update[cj]
                rj = view(centers, :, cj)
                xj = view(x, :, j)
                if cweights[cj] > 0
                    @inbounds rj .+= xj * wj
                else
                    @inbounds rj .= xj * wj
                end
                cweights[cj] += wj
            end
        end
    end

    # sum ==> mean
    for j = 1:k
        if to_update[j]
            @inbounds centers[:, j] .*= 1 / cweights[j]
        end
    end

    # sparse feature selection
    if sparsity < d
        # compute the sparsity criterion
        totalweight = sum(cweights)
        fill!(spcriterion, 0)
        for i in 1:d # features
            for j in 1:k # centers
                spcriterion[i] += cweights[j] * centers[i, j] * centers[i, j]
            end
            spcriterion[i] -= totalweight * globalcenters[i] * globalcenters[i]
        end
        # replace shared feature centers by global feature centers
        fill!(unshared, true)
        sortperm!(spsortidx, spcriterion, initialized=true)
        for i in 1:(d - sparsity)
            unshared[spsortidx[i]] = false
            centers[spsortidx[i], :] .= globalcenters[spsortidx[i]]
        end
    end
end


#
#  Re-picks centers that get no samples assigned to them.
#
function repick_unused_centers(
    x::Matrix{T},           # in: the sample set (d x n)
    costs::Vector{T},       # in: the current assignment costs (n)
    centers::Matrix{T},     # to be updated: the centers (d x k)
    unused::Vector{Int}) where T<:AbstractFloat    # in: the set of indices of centers to be updated

    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(x, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = x[:,j]
        centers[:,i] = v

        colwise!(ds, SqEuclidean(), v, x)
        tcosts = min(tcosts, ds)
    end
end
