module VonMisesFisherModels

using Distributions, Random, StatsBase, LinearAlgebra
import SpecialFunctions: besselix

export
    # Statistical Models
    VonMisesFisherBayesianModel,
    VonMisesFisherMixtureModel,
    VonMisesFisherHiddenMarkovModel,
    VonMisesFisherStateSpaceModel,
    # VonMisesFisherStateSpaceMixtureModel,

    # Returns values of the model parameters in an MCMC chain.
    gibbsInference,
    filterAux,
    backwardSampling,
    smoothingSample

include("SliceSampler.jl")
include("VonMisesFisherBayesianModel.jl")
include("VonMisesFisherMixtureModel.jl")
include("VonMisesFisherHiddenMarkovModel.jl")
include("VonMisesFisherStateSpaceModel.jl")
end
