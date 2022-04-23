struct VonMisesFisherHiddenMarkovModel
    clusterDist::VonMisesFisherBayesianModel
    K::Int
    β::Float64
end

function filtering(Y, θ, μs, κs, α, t, l)
    T = size(Y)[2]
    N = size(μs)[2]

    emission = exp(logpdf(VonMisesFisher(μs[:,l], κs[l]), Y[:,t]))

    #logprobs = zeros(N)
    for k = 1:N
        #logprobs[k] = logvMFpdfSinglex(μs[:,k], κs[k], Y[:,t]) + log(α[t,k])
    end
    #maxLogProb = maximum(logprobs)
    
    divisor = 0
    for k = 1:N
        #divisor += exp(logprobs[k] - maxLogProb)
        divisor += exp(logpdf(VonMisesFisher(μs[:,k], κs[k]), Y[:,t])) * α[t,k]
    end

    #divisor = maxLogProb + log(divisor)
    
    emission * (α[t,l]) / divisor
end

function forwardFiltering(Y, θ, μs, κs)
    T = size(Y)[2]
    N = size(μs)[2]
    
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
                f[t-1,k] = filtering(Y, θ, μs, κs, α, t-1, k)
                #println("res = $((f[t-1,k]))")
                α[t,l] += (θ[k,l]) * f[t-1,k]
            end
        end
    end

    for k = 1:N
        f[T,k] = filtering(Y, θ, μs, κs, α, T, k)
    end
    α, f
end

function forwardFilteringBackwardSampling(Y, θ, μs, κs)
    T = size(Y)[2]
    N = size(μs)[2]

    _, f = forwardFiltering(Y, θ, μs, κs)

    S = zeros(Int64, T)
    S[T] = rand(Categorical(f[T,:]))

    for t = T-1:-1:1
        pS = zeros(N)
        for j = 1:N
            divisor = 0
            for k = 1:N
                divisor += θ[k,S[t+1]] * f[t,k]
            end
            
            pS[j] = θ[j,S[t+1]] * f[t,j] / divisor
        end

        S[t] = rand(Categorical(pS))
    end
    
    S
end

function countTransistions(X, i, j)
    c = 0
    for k = 1:length(X)-1
        if X[k] == i && X[k+1] == j
            c += 1
        end
    end
    c
end

function gibbsInference(model::VonMisesFisherHiddenMarkovModel, Y, niter)
    D = size(Y)[1]
    T = size(Y)[2]
    β = model.β
    N = model.K

    # Transition probability matrix.
    # θ[i,j] = P(xⱼ | xᵢ)
    θ = (ones(N)/N)'
    for i = 2:N
        θ = vcat(θ, (ones(N)/N)')
    end

    # States.
    X = zeros(Int64, T)
    X[1] = rand(Categorical(ones(N)/N))
    for t = 2:T
        #println("t = $(t)")
        X[t] = rand(Categorical(θ[X[t-1],:]))
        #X[t] = rand(Categorical(ones(N)/N))
    end
    
    # Emission Parameters
    # μs[:,n] = nth emission mean direction vector
    μs = zeros(D,N)
    κs = zeros(N)
    for n = 1:N
        Yₙ = Y[:, X .== n]
        #for n2 = 1:size(Yₙ)[2]
        #	μs[:,n] = μs[:,n] + Yₙ[:,n2]
        #end
        #μs[:,n] = μs[:,n] / size(Yₙ)[2]
        μs[:,n], κs[n] = gibbsInference(model.clusterDist, Yₙ, 10)[end]
    end

    #zs = zeros(Int64, niter,T)
    
    # Gibbs sampling
    for i = 1:niter
        #println("\nStarting iteration $(i)")
        
        # Sample θ
        for j = 1:N
            param = zeros(N)
            for k = 1:N
                param[k] = β + countTransistions(X,j,k)
                #param[k] = β + sum(X .== k)
                #param[k] = β + size(X)[1]/N
            end
            θ[j,:] = rand(Dirichlet(param))
        end
        
        # Sample X
        #println("Starting forward backward algorithm")
        # Needs to be forward filtering backward sampling.
        X = forwardFilteringBackwardSampling(Y, θ, μs, κs)
        
        # Sample emission parameters
        #println("Starting emission parameter sampling")
        for n = 1:N
            Yₖ = Y[:, X .== n]
            
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

    (θ, X, μs, κs)
end