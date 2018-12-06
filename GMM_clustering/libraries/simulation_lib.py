import numpy as np

# Draw component indicators from a matrix of probabilities.
# Args:
#   - probs: An n_obs x k matrix of probabilities, where each row
#            sums to one
# Returns:
#   - components: An n_obs-length vector of component indicators
#                 from 0 to (k - 1)
#   - indicator_mat: An n_obs x k matrix of component indicators.
def draw_components(probs):
    n_obs = probs.shape[0]
    u = np.random.random((n_obs, 1))
    return draw_components_from_unif(probs, u)

# Draw component indicators from a matrix of probabilities given a particular
# uniform random draw.
# Args:
#   - probs: See draw_components
#   - u: An n_obs-length vector of uniform random variables.
# Returns:
#   - Same as draw_components but using u to allocate the clusters.  Calling
#     this function multiple times with the same u will always give the same
#     cluster assignments.
def draw_components_from_unif(probs, u):
    n_obs = len(u)
    assert probs.shape[0] == n_obs
    selection = (u <= np.cumsum(probs, axis=1))
    components = np.argmax(selection, axis=1)
    indicator_mat = np.zeros_like(probs)
    indicator_mat[range(n_obs), components] = 1
    return components, indicator_mat

# Simulate data from a mixture model.
# Args:
#   - n_obs: The number of observations.
#   - dim: The dimension of the feature space.
#   - true_k: The true number of clusters.
#   - separation: The ratio of the standard deviation to the mean
#       separation.
# Returns:
#   features (the data) followed by true parameters.
def simulate_data(n_obs, dim, true_k, separation=0.4):
    true_probs = np.linspace(5, 5 + true_k, true_k)
    true_probs /= np.sum(true_probs)

    true_components, true_z = draw_components(
        np.broadcast_to(true_probs, (n_obs, true_k)))

    true_centroids = np.array([
        np.full(dim, k - 0.5 * true_k) for k in range(true_k)])
    true_covs = np.array([
        (separation ** 2) * (k + 1) * \
        np.eye(dim) / true_k for k in range(true_k) ])

    features = np.full((n_obs, dim), float('nan'))
    for n in range(n_obs):
        k = true_components[n]
        features[n, :] = np.random.multivariate_normal(
            true_centroids[k], true_covs[k])

    return features, true_z, true_components, \
        true_centroids, true_covs, true_probs
