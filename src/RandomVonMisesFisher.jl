# Householder transformation
function getHouseHolderVector(x, y)
    v = x - norm(x) * y
    v / norm(v)
end

function rotateToPoint(v, x)
    rmat = 1.0I - 2 * (v * v')
    rmat * x
end

# Wood (1994)
# Algorithm VM*
function genZUW(b, m)
    Z = rand(Beta((m-1)/2, (m-1)/2))
    U = rand(Uniform(0,1))
    W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
    
    Z, U, W
end

function vMFRandWood(μ, λ)
    # Dimension
    m = size(μ)[1]

    # Step 0
    b = (-2λ + sqrt(4 * λ^2 + (m-1)^2)) / (m - 1)
    x0 = (1 - b) / (1 + b)
    c = λ * x0 + (m - 1) * log(1 - x0^2)

    # Step 1 & 2
    Z, U, W = genZUW(b, m)
    while λ * W + (m-1) * log(1 - x0 * W) - c < log(U)
        Z, U, W = genZUW(b, m)
    end

    # Step 3
    V = rand(Normal(0, 1), m-1)
    V = V / norm(V)

    ans = [sqrt(1 - W^2) * V; W]

    tμ = zeros(m)
    tμ[end] = 1.0
    ν = getHouseHolderVector(tμ, μ)
    rotateToPoint(ν, ans)
end

function vMFRandWood(μ, λ, n)
    # Dimension
    m = size(μ)[1]

    X = zeros(m,n)
    for i = 1:n
        X[:,i] = vMFRandWood(μ, λ)
    end
    X
end