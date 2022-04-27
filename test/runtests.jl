using VonMisesFisherModels, LinearAlgebra, Distributions
using Test

@testset "VonMisesFisherModels.jl" begin
    # Write your tests here.

function basicModelTest()
    # Generate some data
    dataμ = [-1, 1, 1] / norm([-1, 1, 1])
    dataκ = 8
    data = rand(VonMisesFisher(dataμ, dataκ), 2000)

    model = VonMisesFisherBayesianModel(VonMisesFisher(ones(size(data)[1]) / norm(ones(size(data)[1])), 0.01), Gamma(1.0,6.0))
    res = gibbsInference(model, data, 1000)
	resμ, resκ = res[1]
	for i = 2:length(res)
		resμ = resμ + res[i][1]
		resκ = resκ + res[i][2]
	end
	println("Basic Bayesian results: $(resμ ./ length(res)), $(resκ / length(res))")
end

function mixtureTest()
    mixdataμ = ([-1, 1, 1] / norm([-1, 1, 1]), [1, -1, -1] / norm([1, -1, -1]))
	mixdataκ = (4.0, 8.0)
	mixdata = (rand(VonMisesFisher(mixdataμ[1], mixdataκ[1]), 1000), rand(VonMisesFisher(mixdataμ[2], mixdataκ[2]), 1000))

    clusterModel = VonMisesFisherBayesianModel(VonMisesFisher(ones(size(mixdata[1])[1]) / norm(ones(size(mixdata[1])[1])), 0.01), Gamma(1.0,6.0))
    mixtureModel = VonMisesFisherMixtureModel(clusterModel, 2, 1.0)
 
    rz, rμ, rκ = gibbsInference(mixtureModel, hcat(mixdata[1], mixdata[2]), 1000)
    println("Mixture results 1: $(rμ[:,1]), $(rκ[1])")
    println("Mixture results 2: $(rμ[:,2]), $(rκ[2])")
end

function mixturePredTest()
    mixdataμ = ([-1, 1, 1] / norm([-1, 1, 1]), [1, -1, -1] / norm([1, -1, -1]))
	mixdataκ = (4.0, 8.0)
	mixdata = (rand(VonMisesFisher(mixdataμ[1], mixdataκ[1]), 1000), rand(VonMisesFisher(mixdataμ[2], mixdataκ[2]), 1000))

    clusterModel = VonMisesFisherBayesianModel(VonMisesFisher(ones(3) / norm(ones(3)), 0.01), Gamma(1.0,6.0))
    mixtureModel = VonMisesFisherMixtureModel(clusterModel, 2, 1.0)
    
    VonMisesFisherModels.fit(mixtureModel, hcat(mixdata[1],mixdata[2]))
    z = predict(mixtureModel, hcat(mixdata[1], mixdata[2]))
    println("$(sum(z .== 1)), $(sum(z .== 2))")
    #println(size(z))
    println("Rand index: $(randIndex(z, vcat(ones(1000), ones(1000)*2)))")
end

function hiddenMarkovModelTest()
    # HMM Parameters
    T = 1000
    N = 2 # Number of clusters
    D = 3

    # Emission Parameters
    # μs[:,i] = mean of emission variable i
    # κs[i] = concentration of emission variable i
    μs = zeros(D, N)
    κs = zeros(N)
    dataμ = ([-1, 1, 1] / norm([-1, 1, 1]), [1, -1, -1] / norm([1, -1, -1]))
    dataκ = (4.0, 8.0)
    for i = 1:N
        μs[:, i] = dataμ[i]
        κs[i] = dataκ[i]
    end

    # Transition probability matrix.
    # π[i,j] = P(xⱼ | xᵢ)
    θ = [0.75 0.25; 0.4 0.6] #hcat(ones(N)/N, ones(N)/N)'

    # Generate data
    Y = Array{Matrix{Float64}}(undef, T)
    X = zeros(Int64, T)
    X[1] = 1
    Y[1] = rand(VonMisesFisher(μs[:, 1], κs[1]), 10)
    for t = 2:T
        X[t] = rand(Categorical(θ[X[t-1], :]))
        Y[t] = rand(VonMisesFisher(μs[:, X[t]], κs[X[t]]), 10)
    end

    clusterModel = VonMisesFisherBayesianModel(VonMisesFisher(ones(D) / norm(ones(D)), 0.01), Gamma(1.0, 6.0))
    hmm = VonMisesFisherHiddenMarkovModel(clusterModel, 2, 1.0)

    θ, X, μs, κs = gibbsInference(hmm, Y, 1000)

    println("θ = $(θ)")
    println("μ1 = $(μs[:,1])")
    println("μ2 = $(μs[:,2])")
    println("κ  = $(κs)")

end

# Generate data
function genDataStateSpace(T)
    NumSamples = 200
    D = 3
    
    # Measurement Variance
    M0 = 40.0
    
    # System Variance
    C0 = 60.0 
    
    Y = Array{Matrix{Float64}}(undef, T)
    for t = 1:T
        #  Y[t] = zeros(NumSamples, D)
        Y[t] = zeros(D, NumSamples)
    end
    #Y = zeros(T,NumSamples,D)
    state = zeros(D,T)
    state[:,1] = [-1, 1, 1] / norm([-1, 1, 1])

    # Generate t=1 samples
    for n = 1:NumSamples
        Y[1][:,n] = rand(VonMisesFisher(state[:,1],M0))
    end

    # Generate remaining dependent samples and states
    for t = 2:T
        state[:,t] = rand(VonMisesFisher(state[:,t-1],C0))
        for i = 1:NumSamples
            Y[t][:,i] = rand(VonMisesFisher(state[:,t],M0))
        end
    end
    
    Y, state
end

function stateSpaceModelTest()
    data, states = genDataStateSpace(20)

    # Filtering
    # states, weights = filterAux(data, 5000, 40, 60)
    
    #fs = rand(VonMisesFisher(states[20,rand(Categorical(weights[20,:])),:], 60))
    #sampledStates = backwardSampling(states, fs, 40)
    #sampledStates = smoothingSample(data, 1000, 40, 60)
    println(states)
    model = VonMisesFisherStateSpaceModel(Gamma(1.0, 6.0), Gamma(1.0, 6.0))
    S, M0, C0 = gibbsInference(model, data, 50)

    println("Measurement κ: $(M0)")
    println("Transition κ: $(C0)")
    println("States: $(S)")
end

# @test_nowarn basicModelTest()
# @test_nowarn mixtureTest()
# @test_nowarn mixturePredTest()
# @test_nowarn hiddenMarkovModelTest()
@test_nowarn stateSpaceModelTest()

end
