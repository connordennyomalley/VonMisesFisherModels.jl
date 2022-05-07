
function logbesseliApprox(ν, a)
    sqrt(a^2 + (ν + 1)^2) + (ν + 0.5) * log(a / (ν + 0.5 + sqrt(a^2 + (ν + 1)^2))) - 0.5 * log(a/2) + (ν + 0.5) * log((2 * ν + 3/2)/(2* (ν + 1))) - 0.5 * log(2 * π)
end

"""
    logC(κ, D)

Log of the constant in the Von Mises-Fisher distribution.
"""
function logC(κ, D)
    # println("kappa: $(κ), D: $(D)")
    # println("V3: $((-D/2) * log((2 * π)) + ((D/2)-1) * log(κ))")
    # println("V4: $((log(besselix(((D/2)-1), κ)) + κ))")
    # (log(besselix(((D/2)-1), κ)) + κ)
    (-D/2) * log((2 * π)) + ((D/2)-1) * log(κ) - logbesseliApprox(((D/2)-1), κ)
end

function logvMFpdfSum(μ, κ, sumX, N)
    D = size(sumX)[1]
    
    # println("V1: $(logC(κ[1], D))")
    # println("V2: $(κ[1] .* (μ' * sumX))")

    N * logC(κ[1], D) + κ[1] .* (μ' * sumX)
end

"""
    logvMFpdf(μ, κ, X)

Log pdf of the Von Mises-Fisher for multiple data points.
"""
function logvMFpdf(μ, κ, X)
    N = size(X)[2]
    D = size(X)[1]

    N * logC(κ[1], D) + κ[1] .* (μ' * sum(X, dims=2)[:,1])
end

"""
    samplePosterior(μ, κ, X, priorDist::VonMisesFisher)

Sample from the posterior of μ for a basic Von Mises-Fisher Bayesian model.
"""
function samplePosteriorμ(κ, sumX, priorDist::VonMisesFisher)
    μ0 = priorDist.μ
    κ0 = priorDist.κ

    νX = κ0 .* μ0 + κ .* sumX
    νXNorm = norm(νX)
    
    # println(νXNorm)
    # println(size(νX))

    vMFRandWood(νX / νXNorm, νXNorm)
    #rand(VonMisesFisher(νX / νXNorm, νXNorm))
end

"""
    posteriorDensityκ(μ, κ, X, priorDist) 

Posterior density of κ for a basic Von Mises-Fisher Bayesian model.
"""
function posteriorDensityκ(μ, κ, sumX, N, priorDist)
    if κ < 0
        return 0
    end

    # println(κ) # 33
    #println(N) # 1001

    loglikelihood = logvMFpdfSum(μ, κ, sumX, N)
    res = loglikelihood + logpdf(priorDist, κ)

    res
end

function samplePosteriorκ(μ, κ, sumX, N, priorDist)
    sliceSample(j -> posteriorDensityκ(μ, j, sumX, N, priorDist), 3, 3, 10, κ)[end]
end

function MLEκ(X)
    R = norm(sum(X, dims=2)[:,1] / size(X)[2])
    p = size(X)[1]
    R * (p - R^2)/(1-R^2)
end

struct VonMisesFisherBayesianModel
    μPrior::VonMisesFisher
    κPrior::ContinuousUnivariateDistribution
end

function gibbsInference(model::VonMisesFisherBayesianModel, X::Matrix, niter::Int)
    # Markov Chain of samples.
    res = Array{Tuple{Vector{Float64},Float64}}(undef, niter + 1)

    # Save recomputing this value
    sumX = sum(X, dims=2)[:,2]

    # Initial Parameter Estimate to start the Markov Chain.
    μ = sumX / size(X)[2]
    κ = MLEκ(X)

    res[1] = (μ, κ)

    for i = 2:niter+1
        μPrevious = res[i-1][1]
        κPrevious = res[i-1][2]

        #println("Sampling μ")
        μSample = samplePosteriorμ(κPrevious, sumX, model.μPrior)

        #println("Sampling κ")
        #println(μSample)
        #println(κPrevious)
        κSample = samplePosteriorκ(μSample, κPrevious, sumX, size(X)[2], model.κPrior)

        res[i] = (μSample, κSample)
        #println(i)
    end

    res[2:end]
end