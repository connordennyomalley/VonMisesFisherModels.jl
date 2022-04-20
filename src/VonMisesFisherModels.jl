module VonMisesFisherModels
using Distributions, Random, StatsBase, LinearAlgebra
import SpecialFunctions: besselix

export
    # Statistical Models
    VonMisesFisherBayesianModel,
#    VonMisesFisherMixtureModel,
#    VonMisesFisherHiddenMarkovModel,
#    VonMisesFisherStateSpaceModel,

    # Returns values of the model parameters in an MCMC chain.
    gibbsInference

include("SliceSampler.jl")
include("VonMisesFisherBayesianModel.jl")
end
