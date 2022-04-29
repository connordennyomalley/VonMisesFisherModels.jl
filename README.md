# VonMisesFisherModels
Notation used in library:
- Uppercase letters represent matrices of multiple observations, where the first dimension is the dimensions of the data and the second dimension is the number of observations. e.g. $X \in \mathbb{R}^{D \times N}$ is a matrix with $D$ rows and $N$ columns. In julia `X = zeros(D,N)` can initialize the matrix and `X[:,i]` is the `i`th observation.
- Anyname can be used for constants or singular values, including uppercase and lower case letters. e.g. `D` or `N` from above. The meaning of variables is context dependent.
- A single lowercase letter is used to represent a single data point. e.g. if `X[:,i]` is being passed in to a function, the functions formal parameter will be labeled lowercase, to indicate single observation `x`.
- `S` refers to data that represents state. E.g. in Hidden Markov Models or State Space models.
- `X` represents any series of data, not necessarily observation or state data. Mainly used in general probability density functions where the meaning of the data means nothing other than the series being jointly distributed and i.i.d. from some distribution.
- `Y` represnets time series data, and `Y[i]` is the observations at time interval `i`, which will be the same form as `X`
