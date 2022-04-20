
export sliceSample

# Stepping out approach to finding the interval
function findInterval(f, x0, y, ω, m)		
	# Process
	U = rand(Uniform(0,1))
	L = x0 - ω*U
	R = L + ω
	V = rand(Uniform(0,1))
	J = floor(m*V)
	K = (m - 1) - J

	while J > 0 && y < f(y)
		L = L - ω
		J = J - 1
	end
	while K > 0 && y < f(R)
		R = R + ω
		K = K - 1
	end

	if L < 0
		L = 0
	end
	if R < 0
		R = 0
	end
	return (L,R)
end

# Determines if x1 is in an acceptable next state for the Markov Chain.
# Not required unless using doubling approach to find interval.
function accept(f, x0, x1, y, ω, L, R)
	Lhat = L
	Rhat = R
	D = false

	while Rhat - Lhat > 1.1 * ω
		M = (Lhat + Rhat) / 2
		
		if (x0 < M && x1 >= M) || (x0 >= M && x1 <= M)
			D = true
		end

		if x1 < M
			Rhat = M
		else
			Lhat = M
		end

		if D && y >= f(Lhat) && y >= f(Rhat)
			return false
		end
		
	end

	return true
end

# Samples a new point from a given slice.
function shrinkageSample(f, x0, y, ω, L, R)
	Lbar = L
	Rbar = R
	
	while true
		U = rand(Uniform(0,1))
		x1 = Lbar + U * (Rbar - Lbar)

		if y < f(x1) #&& accept(f, x0, x1, y, ω, L, R)
			return x1
		end
		
		if x1 < x0
			Lbar = x1
		else
			Rbar = x1
		end
		
	end
end


function sliceSample(f, ω, m, n, κInit=1, isLog=true)
	res = Array{Float64}(undef, n)
	res[1] = κInit # Set this to value from X?

	distances = Array{Float64}(undef, n - 1)
	
	for i = 2:length(res)
		x0 = res[i-1]

		if isLog
			z = f(x0) - rand(Exponential(1))
			I = findInterval(f, x0, z, ω, m)
			res[i] = shrinkageSample(f, x0, z, ω, I[1], I[2])
		else
			y = rand(Uniform(0.0,f(x0)))
			I = findInterval(f, x0, y, ω, m)
			res[i] = shrinkageSample(f, x0, y, ω, I[1], I[2])
		end

		# Update omega to be closer to a reasonable value
		#distances[i-1] = abs(x0 - res[i])
		#ω = distances[i-1]/i
		
	end

	res
end