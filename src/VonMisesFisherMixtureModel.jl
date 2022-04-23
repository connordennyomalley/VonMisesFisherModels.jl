
struct VonMisesFisherMixtureModel
    clusterDist::VonMisesFisherBayesianModel
    K::Int
    β::Float64
end

function gibbsInference(model::VonMisesFisherMixtureModel, X::Matrix, niter::Int)
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
    for kᵢ = 1:K
        Xₖ = X[:,z .== kᵢ]
        μs[:,kᵢ], κs[kᵢ] = gibbsInference(model.clusterDist, Xₖ, 100)[end]
    end

    for i = 1:niter

        # Sample phi
        param = zeros(length(ϕ))
        for j = 1:length(ϕ)
            param[j] = β + sum(z .== j)
        end
        ϕ = rand(Dirichlet(param))

        # Sample z
        for j = 1:N
            xᵢ = X[:, j]

            # Construct probability vector over the clusters.
            pvec = zeros(K)
            for kᵢ = 1:K
                pvec[kᵢ] = exp(log(ϕ[kᵢ]) + logpdf(VonMisesFisher(μs[:, kᵢ], κs[kᵢ]), vec(xᵢ)))
            end
            pvec = pvec / sum(pvec)

            z[j] = rand(Categorical(pvec))
        end

        # Sample parameters for each cluster k
        for kᵢ = 1:K
            Xₖ = X[:, z.==kᵢ]
            if size(Xₖ)[2] > 0
                # Alternative MLE method
                # (sum(Xₖ, dims=2)[:,1] / size(Xₖ)[2], MLEκ(Xₖ))
                μs[:, kᵢ], κs[kᵢ] = gibbsInference(model.clusterDist, Xₖ, 100)[end]
            else
                # Leave the same as old volue for cluster parameters
            end
        end
    end
    (z,μs,κs)
end
