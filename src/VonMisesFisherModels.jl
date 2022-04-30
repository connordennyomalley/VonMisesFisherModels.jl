module VonMisesFisherModels

using Distributions, Random, StatsBase, LinearAlgebra
import SpecialFunctions: besselix

export
    ## Statistical Models
    VonMisesFisherBayesianModel,
    VonMisesFisherMixtureModel,

    ## Temporal Models
    VonMisesFisherHiddenMarkovModel,
    VonMisesFisherStateSpaceModel,
    # VonMisesFisherStateSpaceMixtureModel,

    ## Metrics
    randIndex,
    
    ## Training and Prediction
    predict,
    predictProba,
    fit,
    gibbsInference,
    # filterAux,
    # backwardSampling,
    smoothingSample,

    ## Utility functions
    vMFRandWood,
    tfidf
    
include("TfIdf.jl")
include("RandomVonMisesFisher.jl")
include("Metrics.jl")
include("SliceSampler.jl")
include("VonMisesFisherBayesianModel.jl")
include("VonMisesFisherMixtureModel.jl")
include("VonMisesFisherHiddenMarkovModel.jl")
include("VonMisesFisherStateSpaceModel.jl")
end
