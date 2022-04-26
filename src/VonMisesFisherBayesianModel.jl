"""
    logC(κ, D)

Log of the constant in the Von Mises-Fisher distribution.
"""
function logC(κ, D)
    log((2 * π)^(-D/2)) + log(κ^((D/2)-1)) - (log(besselix(((D/2)-1), κ)) + κ)
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
function samplePosteriorμ(μ, κ, X, priorDist::VonMisesFisher)
    μ0 = priorDist.μ
    κ0 = priorDist.κ
    #  println("")
    #println(size(X))
    #println(κ0)
    #println(μ0)
    #println(size(κ0 .* μ0))
    #println(size(κ .* sum(X, dims=2)))
    νX = κ0 .* μ0 + κ .* sum(X, dims=2)[:,1]
    νXNorm = norm(νX)
    #println("$(νX)")
    rand(VonMisesFisher(νX / νXNorm, νXNorm))
end

"""
    posteriorDensityκ(μ, κ, X, priorDist) 

Posterior density of κ for a basic Von Mises-Fisher Bayesian model.
"""
function posteriorDensityκ(μ, κ, X, priorDist)
    if κ < 0
        return 0
    end

    loglikelihood = logvMFpdf(μ, κ, X)
    loglikelihood + logpdf(priorDist, κ)
end

function samplePosteriorκ(μ, κ, X, priorDist)
    sliceSample(j -> posteriorDensityκ(μ, j, X, priorDist), 3, 3, 10, κ)[end]
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

    # Initial Parameter Estimate to start the Markov Chain.
    μ = sum(X, dims=2)[:,1] / size(X)[2]
    κ = MLEκ(X)

    res[1] = (μ, κ)

    for i = 2:niter+1
        μPrevious = res[i-1][1]
        κPrevious = res[i-1][2]

        μSample = samplePosteriorμ(μPrevious, κPrevious, X, model.μPrior)
        κSample = samplePosteriorκ(μSample, κPrevious, X, model.κPrior)

        res[i] = (μSample, κSample)
    end

    res[2:end]
end