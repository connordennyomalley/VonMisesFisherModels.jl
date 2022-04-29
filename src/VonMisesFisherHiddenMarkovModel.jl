mutable struct VonMisesFisherHiddenMarkovModel
    clusterDist::VonMisesFisherBayesianModel
    K::Int
    β::Float64

    # Fit and prediction parameters
    μs
    κs
    θ

    function VonMisesFisherHiddenMarkovModel(clusterDist::VonMisesFisherBayesianModel, K::Int, β::Float64)
        new(clusterDist, K, β)
    end
end

function fit(model::VonMisesFisherHiddenMarkovModel, data::AbstractArray)
    # Return states for unsupervised learning.
    model.θ, S, model.μs, model.κs = gibbsInference(model, data, 100)
    S
end

function predict(model::VonMisesFisherHiddenMarkovModel, data::AbstractArray)
    # data is a series of observations and we want to predict from which cluster they came.
    # ignoring the transistion distributions.
    probs = predictProba(model, data)
    
    res = zeros(size(probs)[2])
    for i = 1:size(probs)[2]
        _, res[i] = findmax(probs[:,i])
    end

    res
end

function predictProba(model::VonMisesFisherHiddenMarkovModel, data::AbstractArray)
    probs = zeros(model.K, size(data)[2])

    for n = 1:size(data)[2]
        # Construct probability vector over the clusters.
        pvec = zeros(model.K)
        for kᵢ = 1:model.K
            pvec[kᵢ] = exp(logpdf(VonMisesFisher(model.μs[:, kᵢ], model.κs[kᵢ]), data[:,n]))
        end
        pvec = pvec / sum(pvec)

        probs[:,n] = pvec
    end

    probs
end

function predictProba(model::VonMisesFisherHiddenMarkovModel, data::AbstractArray, currentCluster::Int)
    probs = zeros(model.K, size(data)[2])

    for n = 1:size(data)[2]
        # Construct probability vector over the clusters.
        pvec = zeros(model.K)
        for kᵢ = 1:model.K
            pvec[kᵢ] = exp(log(model.θ[currentCluster, kᵢ]) + logpdf(VonMisesFisher(model.μs[:, kᵢ], model.κs[kᵢ]), data[:,n]))
        end
        pvec = pvec / sum(pvec)

        probs[:,n] = pvec
    end

    probs
end

function filtering(Y, μs, κs, α, t, l)
    emission = logvMFpdf(μs[:,l], κs[l], Y[t])

    ((emission) + log(α[t,l]))
end

function forwardFiltering(Y, θ, μs, κs)
    T = size(Y)[1]
    N = size(μs)[2]
    
    # α is the one step ahead density.
    α = zeros(T,N)
    for l = 1:N
        for k = 1:N
            α[1,l] += θ[k,l] * (1/N)
        end
    end

    f = zeros(T,N)
    
    for t = 2:T
        for l = 1:N
            for k = 1:N
                # This is the log! remember needs normalising
                f[t-1,k] = filtering(Y, μs, κs, α, t-1, k)
            end
            f[t-1,:] = exp.(f[t-1,:] .- maximum(f[t-1,:]))
            f[t-1,:] = f[t-1,:] / sum(f[t-1,:])

            for k = 1:N
                
                #println("res = $((f[t-1,k]))")
                α[t,l] += (θ[k,l]) * (f[t-1,k])
            end
        end
    end

    for k = 1:N
        f[T,k] = filtering(Y, μs, κs, α, T, k)
    end
    f[T,:] = exp.(f[T,:] .- maximum(f[T,:]))
    f[T,:] = f[T,:] / sum(f[T,:])

    α, f
end

function forwardFilteringBackwardSampling(X, θ, μs, κs)
    T = size(X)[1]
    K = size(μs)[2]

    _, f = forwardFiltering(X, θ, μs, κs)

    S = zeros(Int64, T)
    S[T] = rand(Categorical(f[T,:]))

    for t = T-1:-1:1
        pS = zeros(K)
        for j = 1:K
            divisor = 0
            for k = 1:K
                divisor += θ[k,S[t+1]] * f[t,k]
            end
            
            pS[j] = θ[j,S[t+1]] * f[t,j] / divisor
        end

        S[t] = rand(Categorical(pS))
    end
    
    S
end

"""
    countTransitions(S, i, j)

Returns given a path through states how many transitions there are from state i to state j.
"""
function countTransistions(S, i, j)
    c = 0
    for k = 1:length(S)-1
        if S[k] == i && S[k+1] == j
            c += 1
        end
    end
    c
end

function gibbsInference(model::VonMisesFisherHiddenMarkovModel, Y, niter)
    D = size(Y[1])[1]
    T = size(Y)[1]
    β = model.β
    K = model.K

    # Transition probability matrix.
    # θ[i,j] = P(xⱼ | xᵢ)
    θ = (ones(K)/K)'
    for i = 2:K
        θ = vcat(θ, (ones(K)/K)')
    end

    # States.
    S = zeros(Int64, T)
    S[1] = rand(Categorical(ones(K)/K))
    for t = 2:T
        #println("t = $(t)")
        S[t] = rand(Categorical(θ[S[t-1],:]))
        #X[t] = rand(Categorical(ones(N)/N))
    end
    
    # Emission Parameters
    # μs[:,n] = nth emission mean direction vector
    μs = zeros(D,K)
    κs = zeros(K)
    for n = 1:K
        Yₙ = reduce(hcat, Y[S .== n])
        μs[:,n], κs[n] = gibbsInference(model.clusterDist, Yₙ, 10)[end]
    end

    #zs = zeros(Int64, niter,T)
    
    # Gibbs sampling
    for i = 1:niter
        #println("\nStarting iteration $(i)")
        
        # Sample θ
        for j = 1:K
            param = zeros(K)
            for k = 1:K
                param[k] = β + countTransistions(S,j,k)
                #param[k] = β + sum(X .== k)
                #param[k] = β + size(X)[1]/N
            end
            θ[j,:] = rand(Dirichlet(param))
        end
        
        # Sample X
        #println("Starting forward backward algorithm")
        # Needs to be forward filtering backward sampling.
        S = forwardFilteringBackwardSampling(Y, θ, μs, κs)
        
        # Sample emission parameters
        #println("Starting emission parameter sampling")
        for n = 1:K
            Yₖ = reduce(hcat, Y[S .== n])
            
            if size(Yₖ)[2] > 1
                # (sum(Xₖ, dims=2)[:,1] / size(Xₖ)[2], MLEκ(Xₖ))
                #μs[:,n], κs[n] = (sum(Yₖ, dims=2)[:,1] / size(Yₖ)[2], MLEκ(Yₖ))
                #println("num data points = $(size(Yₖ)[2])")
                #println("μ = $(μs[:,n])")
                #println("κ = $(κs[n])")
                μs[:,n], κs[n] = gibbsInference(model.clusterDist, Yₖ, 10)[end]
            else
                #μs[:,n] = Yₖ
            end
        end
    end

    (θ, S, μs, κs)
end