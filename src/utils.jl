# Common utilities

##### common types

abstract type ClusteringResult end

# generic functions

nclusters(R::ClusteringResult) = length(R.counts)
counts(R::ClusteringResult) = R.counts
assignments(R::ClusteringResult) = R.assignments
nunsharedfeatures(R::ClusteringResult) = count(R.unshared)
nsharedfeatures(R::ClusteringResult) = length(R.unshared) - count(R.unshared)

##### convert weight options

conv_weights(::Type{T}, n::Int, w::Nothing) where {T} = nothing

function conv_weights(::Type{T}, n::Int, w::Vector) where T
    length(w) == n || throw(DimensionMismatch("Incorrect length of weights."))
    convert(Vector{T}, w)::Vector{T}
end

##### convert display symbol to disp level

display_level(s::Symbol) =
    s == :none ? 0 :
    s == :final ? 1 :
    s == :iter ? 2 :
    error("Invalid value for the option 'display'.")


##### update minimum value

function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end
