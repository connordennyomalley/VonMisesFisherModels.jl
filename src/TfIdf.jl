
"""
    tfIdfNorm(documents::AbstractArray{AbstractString})

Returns a mapping of the documents in Tf-Idf form.
"""
function tfIdfNorm(documents::AbstractArray{AbstractString})
    termDocCounts = Dict{AbstractString, Int}()
    N = length(documents)

    for doc = documents
        seen = Set()

        for wordmatch = eachmatch(r"[a-zA-Z]+", doc)
            termkey = lowercase(wordmatch.match)

            # Increase document count for this term
            if !(termkey in seen)
                termDocCounts[termkey] = get(termDocCounts, termkey, 0) + 1
            end

            push!(seen, termkey)
        end
    end

    # Define a mapping from term key to row
    rowmap = Dict(collect(zip(collect(keys(termDocCounts)), collect(1:length(termDocCounts)))))


    mappings = zeros(length(termDocCounts), N)

    i = 1
    for doc in documents
        # Count of terms in current document
        tc = Dict{AbstractString, Int}()
        for wordmatch = eachmatch(r"[a-zA-Z]+", doc)
            termkey = lowercase(wordmatch.match)
            tc[termkey] = get(tc, termkey, 0) + 1
        end

        # the tf-idf value of each term in this document
        for termkey = keys(tc)
            tf = tc[termkey] / length(tc)

            idf = log(N / (1 + termDocCounts[termkey]))

            mappings[rowmap[termkey], i] = tf * idf
        end

        i += 1
    end

    mappings
end