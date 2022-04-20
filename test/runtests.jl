using VonMisesFisherModels, LinearAlgebra, Distributions
using Test

@testset "VonMisesFisherModels.jl" begin
    # Write your tests here.

function ultimateTest()
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
	println((resμ ./ length(res), resκ / length(res)))
end

@test_nowarn ultimateTest()

end
