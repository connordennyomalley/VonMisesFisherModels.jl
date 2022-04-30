
"""
    VonMisesFisherMixtureModel(clusterDist::VonMisesFisherBayesianModel, K::Int, β::Float64)

Represents a basic VMF Mixture Model.
"""
mutable struct VonMisesFisherMixtureModel
    clusterDist::VonMisesFisherBayesianModel
    K::Int
    β::Float64

    # Trained values
    μs::AbstractArray
    κs::AbstractArray{Float64}
    ϕ::AbstractArray

    function VonMisesFisherMixtureModel(clusterDist::VonMisesFisherBayesianModel, K::Int, β::Float64)
        new(clusterDist, K, β)
    end
end

function fit(model::VonMisesFisherMixtureModel, data::AbstractArray)
    # Returns z so model can be used for unsupervised learning.
    z, model.μs, model.κs, model.ϕ = gibbsInference(model, data, 100)
    z
end

function predict(model::VonMisesFisherMixtureModel, data::AbstractArray)
    probs = predictProba(model, data)
    
    res = zeros(size(probs)[2])
    for i = 1:size(probs)[2]
        _, res[i] = findmax(probs[:,i])
    end

    res
end

function predictProba(model::VonMisesFisherMixtureModel, data::AbstractArray)
    probs = zeros(model.K, size(data)[2])

    for n = 1:size(data)[2]
        # Construct probability vector over the clusters.
        pvec = zeros(model.K)
        for kᵢ = 1:model.K
            pvec[kᵢ] = exp(log(model.ϕ[kᵢ]) + logpdf(VonMisesFisher(model.μs[:, kᵢ], model.κs[kᵢ]), data[:,n]))
        end
        pvec = pvec / sum(pvec)

        probs[:,n] = pvec
    end

    probs
end

"""
    gibbsInference(model::VonMisesFisherMixtureModel, X::Matrix, niter::Int) 

Generates an MCMC Chain of parameter sampled values.
"""
function gibbsInference(model::VonMisesFisherMixtureModel, X::Matrix, niter::Int, clusterNIter::Int=5)
    checkInputMixture(X)

    D = size(X)[1]
    N = size(X)[2]
    β = model.β
    K = model.K

    # Mixture parameters init
    ϕ = vec(ones(1, K)/K)
    z = rand(Categorical(ϕ), N)

    # Parameters for each cluster.
    κs = Array{Float64}(undef, K)
    μs = Array{Float64}(undef, D, K)
    #println("Starting gibbs for initial clusters...")
    for kᵢ = 1:K
        Xₖ = X[:,z .== kᵢ]
        μs[:,kᵢ], κs[kᵢ] = gibbsInference(model.clusterDist, Xₖ, clusterNIter)[end]
    end
    #println("Done.")

    for i = 1:niter

        #println("Here!")
        # Sample phi
        param = zeros(length(ϕ))
        for j = 1:length(ϕ)
            param[j] = β + sum(z .== j)
        end
        ϕ = rand(Dirichlet(param))

        #println("Here 2!")
        # Sample z
        for j = 1:N
            xᵢ = X[:, j]

            # Construct probability vector over the clusters.
            pvec = zeros(K)
            for kᵢ = 1:K
                pvec[kᵢ] = (log(ϕ[kᵢ]) + logvMFpdfSum(μs[:, kᵢ], κs[kᵢ], vec(xᵢ), 1))#logpdf(VonMisesFisher(μs[:, kᵢ], κs[kᵢ]), vec(xᵢ))
            end #TODO: NORMALIZE! ✓
            pvec = exp.(pvec .- maximum(pvec))
            pvec = pvec / sum(pvec)

            z[j] = rand(Categorical(pvec))
        end

        
        #println("Here 3!")
        # Sample parameters for each cluster k
        for kᵢ = 1:K
            Xₖ = X[:, z.==kᵢ]
            if size(Xₖ)[2] > 0
                # Alternative MLE method
                # μs[:, kᵢ], κs[kᵢ] = (sum(Xₖ, dims=2)[:,1] / size(Xₖ)[2], MLEκ(Xₖ))

                μs[:, kᵢ], κs[kᵢ] = gibbsInference(model.clusterDist, Xₖ, clusterNIter)[end]
            else
                # Leave the same as old value for cluster parameters
            end
        end
        #println("$(i) ✓")
    end

    (z,μs,κs,ϕ)
end

function checkInputMixture(X)

    # Check data points are not nan
    if sum(isnan.(X)) != 0
        throw(error("Data contains NaN values."))
    end

    # Check data points are normal.
    # for i = 1:size(X)[2]
    #     if norm(X[:,i]) != 1.0
    #         throw(error("Data contains non-normalized vectors. Vectors must have norm of 1."))
    #     end
    # end

end