
function tfidf(corpus; minDf::Float64=0.0, maxDf::Float64=1.0)
    ## Clean data
    D = []
    for doc = corpus
        d = []
        for m = eachmatch(r"\b(\w+)\b", doc)
            push!(d, lowercase(m.match))
        end
        push!(D, d)
    end
    
    ## Term frequencies within documents
    docCountMaps = []
    for d = D
        push!(docCountMaps, countmap(d))
    end

    ## Compute df of words
    documentCounts = Dict{String, Int}()
    for docCountMap = docCountMaps
        for w = keys(docCountMap)
            if docCountMap[w] > 0
                documentCounts[w] = get(documentCounts, w, 0) + 1
            end
        end
    end

    
    ## Filter terms that appear in high percentage of documents
    # Find all terms that need removing
    toRemove = []
    for w = keys(documentCounts)
        docProportion = documentCounts[w]/length(D)
        if docProportion > maxDf || docProportion < minDf
            push!(toRemove, w)
        end
    end

    # Convert to set for quick lookup
    toRemove = Set(toRemove)

    # Remove from dicts
    for dcm = docCountMaps
        filter!(p -> p.first in toRemove, dcm)
    end
    filter!(p -> p.first in toRemove, documentCounts)


    ## Map each document to vector

    # X[:,j] is the vector of the j'th observation
    X = zeros(length(documentCounts), length(corpus))

    # Term index
    i = 1

    # Document index
    j = 1

    for t = keys(documentCounts)
        for cmap = docCountMaps
            # Frequency of term in document
            ftd = get(cmap, t, 0)

            # Words in document
            dMag = sum(collect(values(cmap)))

            # Documents in corpus
            N = length(docCountMaps)

            # Document frequency of term
            dft = documentCounts[t]

            X[i,j] = (ftd / dMag) * (log(10, N/(dft)))

            j += 1
        end
        j = 1
        i += 1
    end
    
    # Return matrix and the labels of the rows
    X, collect(keys(documentCounts))
end