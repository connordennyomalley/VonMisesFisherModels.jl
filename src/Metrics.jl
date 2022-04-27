
function randIndex(X, Y)
    N = size(X)[1]

    a = 0
    b = 0

    # Loop through pairs
    for i = 1:N-1
        for j = i+1:N
            if X[i] == X[j] && Y[i] == Y[j]
                a += 1
            elseif X[i] != X[j] && Y[i] != Y[j]
                b += 1
            end
        end
    end

    (a + b) / (N * (N-1)/2)
end