struct VonMisesFisherStateSpaceModel
    transitionPrior::ContinuousUnivariateDistribution
    emissionPrior::ContinuousUnivariateDistribution
end

function gibbsInference(model::VonMisesFisherStateSpaceModel, Y::Vector{Matrix{Float64}}, niter::Int; numParticles::Int=1000)
    checkInputTemporal(Y)
    
    ## Initial Samples
        
    # Initial states estimate
    S = map(sX -> sum(sX, dims=2) ./ size(sX)[2], Y)
    S = map(sX -> sX ./ norm(sX), S)
    S = reduce(hcat, S)
    
    # M0 = Emission concentration
    M0 = emissionκGibbs(S, Y, model.emissionPrior)[end]

    # Transition concentration
    C0 = transitionκGibbs(S, model.transitionPrior; niter=100)[end]

    #C0chain = zeros(niter)

    #println("Beginning Gibbs Sampling...")
    for i = 2:(niter + 1)
        # Sample path through states
        S = forwardFilteringStateSpaceSample(Y, numParticles, M0, C0)

        # Sample parameters of transition distribution
        C0 = transitionκGibbs(S, model.transitionPrior; niter=100)[end]
        #C0chain[i-1] = C0

        # Sample parameters of emission distribution
        M0 = emissionκGibbs(S, Y, model.emissionPrior)[end]
        #println("$(i-1)/$(niter) ✅")
    end
    
    (S, M0, C0)
end

function logpdfEmissionκ(X, Y, prior, κ)
    if κ < 0
        return -Inf
    end

    # Prob of seeing data under κ then times prior of κ
    v = 0.0
    for t = 1:length(Y)
        v += logpdfX(Y[t], X[:,t], κ)
    end
    v += logpdf(prior, κ)

    v
end

function emissionκGibbs(X, Y, prior; niter::Int=100)
    κ = zeros(niter)
    κ[1] = 1.0
    # println("State size $(size(X))")
    # println("Data size $(size(Y[1]))")

    for i = 2:niter
        κ[i] = sliceSample(v -> logpdfEmissionκ(X, Y, prior, v), 3, 3, 10, κ[i-1])[end]
    end
    κ
end

function logpdfX(X, μ, κ)
    D = size(X)[1]
    N = size(X)[2]
    
    val = 0
    for i = 1:N
        # println(μ)
        if norm(μ) != 1.0
            μ = μ / norm(μ)
        end
        val += logpdf(VonMisesFisher(μ,κ), X[:,i])
    end
    (val)
end

function pdfX(X, μ, κ)
    exp(logpdfX(X,μ,κ))
end

## Auxilary Particle Filter
function filterAux(Y, numParticles, M0, C0)
    T = size(Y)[1]
    D = size(Y[1])[1]

    N = numParticles

    # Resampling threshold
    Nthr = 0.0000001
    
    # State particles
    α = zeros(T,D,N)

    # State particle weights
    w = zeros(T,N)

    # Initial particle weights and values
    model = VonMisesFisherBayesianModel(VonMisesFisher(ones(D) / norm(ones(D)), 0.01), Gamma(1.0,6.0))

    α0 = gibbsInference(model, Y[1], 100)[end][1]
    for k = 1:N
        α[1,:,k] = rand(VonMisesFisher(α0, C0))
        w[1,k] = log(1.0/N) + logpdfX(Y[1], α[1,:,k], M0)
    end
    w[1,:] = exp.(w[1,:] .- maximum(w[1,:]))
    w[1,:] = w[1,:] / sum(w[1,:])

    ## Propagate particles
    for t = 1:T-1
        g = zeros(N)
        μ = zeros(D,N)
        for i = 1:N
            # Estimate of x[t+1]
            μ[:,i] = α[t,:,i]
            μ[:,i] = μ[:,i] / norm(μ[:,i])

            g[i] = log(w[t,i]) + logpdfX(Y[t+1], μ[:,i], M0)
            
        end
        g = exp.(g .- maximum(g))
        g = g / sum(g)

        # Sample
        for i = 1:N
            # Auxiliary Indicator
            j = rand(Categorical(g))
            if norm(α[t,:,j]) != 1.0
                α[t,:,j] = α[t,:,j] / norm(α[t,:,j])
            end
            α[t+1,:,i] = rand(VonMisesFisher(α[t,:,j], C0))

            w[t+1,i] = logpdfX(Y[t+1], α[t+1,:,i], M0) - logpdfX(Y[t+1], μ[:,j], M0)
        end
        w[t+1,:] = exp.(w[t+1,:] .- maximum(w[t+1,:]))
        w[t+1,:] = w[t+1,:] / sum(w[t+1,:])

        # Resampling
        Neff = 1 / sum(w[t+1,:] .^ 2)
        #if Neff < Nthr
        if Neff < Nthr
            for i = 1:N
                v = rand(Categorical(w[t+1,:]))
                α[t+1,:,i] = rand(VonMisesFisher(α[t+1,:,v], C0))
            end
            w[t+1,:] = ones(N) / N
        end
    end
    
    α,w
end

function PLS(X, filterSample, C0)
    # X are filtering particles.
    
    T = size(X)[1]
    N = size(X)[3]
    D = size(X)[2]

    # Weights
    w = zeros(T,N)
    sX = zeros(D,T)
    
    # Initial sample to work backwards from
    sX[:,T] = filterSample
    
    for t = T-1:-1:1
        for i = 1:N
            w[t,:] = zeros(N)
            for j = 1:N
                if norm(X[t,:,i]) != 1.0
                    X[t,:,i] = X[t,:,i] / norm(X[t,:,i])
                end
                w[t,j] = pdf(VonMisesFisher(X[t,:,i], C0), X[t+1,:,i])
            end
            w[t,:] = w[t,:] / sum(w[t,:])

            sX[:,t] = X[t, :, rand(Categorical(w[t,:]))]
        end
    end

    sX
end

function backwardSampling(X, fw, filterSample, C0)
    # X are filtering particles.
    
    T = size(X)[1]
    N = size(X)[3]
    D = size(X)[2]
    
    # Initial sample to work backwards from
    sX = zeros(D,T)
    sX[:,T] = filterSample
    
    for t = T-1:-1:1

        # Compute probability over states for this time step and this particle i
        pvec = zeros(N)
        for j = 1:N
            # Probability of being in the state at t+1
            # probs going from state X[t,:,j] -> sX[:,t+1]
            # X[t,:,j] is the particle under consideration.
            ξ = logpdf(VonMisesFisher(X[t,:, j], C0), sX[:,t+1])

            # Filtering
            fv = 0
            for k = 1:N
                if j != k
                    fv += log(fw[t,k]) + logpdf(VonMisesFisher(X[t,:,k], C0), X[t,:,j])
                end
            end

            pvec[j] = ξ + fv
        end
        pvec = exp.(pvec .- maximum(pvec))
        pvec = pvec / sum(pvec)

        sX[:,t] = X[t,:,rand(Categorical(pvec))]

    end

    sX
end

function forwardFilteringStateSpaceSample(Y, numParticles, M0, C0)
    checkInputTemporal(Y)

    T = size(Y)[1]
    D = size(Y[1])[1]

    # Particle Filtering
    X, fw = filterAux(Y, numParticles, M0, C0)

    # Sample state at t=T
    # fs = rand(VonMisesFisher(X[T,:,rand(Categorical(fw[T,:]))], C0))
    
    S = zeros(D,T)
    for t = T:-1:1
       S[:,t] = rand(VonMisesFisher(X[t,:,rand(Categorical(fw[t,:]))], C0))
       S[:,t] = S[:,t] / norm(S[:,t])
    end
    return S

    # Backward sampling using particles
    # backwardSampling(X, fw, fs, C0)
end

function logPdfκ(κ, x, prior)
    if κ < 0
        return -Inf
    end
    
    T = size(x)[2]

    #println(κ)
    if κ < 0
        return 0
    end
    
    v = 0
    for t = 2:T
        #v *= log(pdf(VonMisesFisher(x[t-1,:], κ),x[t,:]))
        #v += (vMFpdf2(x[t-1,:], κ, x[t,:]))
        x[:,t-1] = x[:,t-1] / norm(x[:,t-1])
        v += logpdf(VonMisesFisher(x[:,t-1], κ), x[:,t])#logvMFpdfSum(x[:,t-1], κ, x[:,t], 1)
        #logvMFpdfSingle(x[t-1,:], κ, x[t,:])
    end

    v += logpdf(prior,κ)
    v
end

function transitionκGibbs(x, prior; niter::Int=100)
    κ = zeros(niter)
    κ[1] = 1.0
    
    for i = 2:niter
        κ[i] = sliceSample(v -> logPdfκ(v, x, prior), 3, 3, 10, κ[i-1])[end]
    end
    κ
end