import autograd
import autograd.numpy as np
import autograd.scipy as sp

import modeling_lib as modeling_lib
import functional_sensitivity_lib as fun_sens_lib
import cluster_quantities_lib as cluster_lib

import paragami

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(dim, k_approx):
    """
    Returns a paragami patterned dictionary
    that stores the variational parameters.

    Parameters
    ----------
    dim : int
        Dimension of the datapoints.
    k_approx : int
        Number of components in the model.

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # cluster centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(dim, k_approx))

    # BNP sticks
    # variational distribution for each stick is logitnormal
    vb_params_paragami['stick_propn_mean'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    vb_params_paragami['stick_propn_info'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)

    # cluster covariances
    vb_params_paragami['gamma'] = \
        paragami.pattern_containers.PatternArray(shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami


##########################
# Set up prior parameters
##########################
def get_default_prior_params(dim):
    """
    Returns a paragami patterned dictionary
    that stores the prior parameters.

    Default prior parameters are those set for the experiments in
    "Evaluating Sensitivity to the Stick Breaking Prior in
    Bayesian Nonparametrics"
    https://arxiv.org/abs/1810.06587

    Parameters
    ----------
    dim : int
        Dimension of the datapoints.

    Returns
    -------
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    prior_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the prior parameters.

    """

    prior_params_dict = dict()
    prior_params_paragami = paragami.PatternDict()

    # DP prior parameter
    prior_params_dict['alpha'] = np.array([3.0])
    prior_params_paragami['alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the centroids
    prior_params_dict['prior_centroid_mean'] = np.array([0.0])
    prior_params_paragami['prior_centroid_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_centroid_info'] = np.array([0.1])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the variance
    prior_params_dict['prior_gamma_df'] = np.array([8.0])
    prior_params_paragami['prior_gamma_df'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_gamma_inv_scale'] = 0.62 * np.eye(dim)
    prior_params_paragami['prior_gamma_inv_scale'] = \
        paragami.PSDSymmetricMatrixPattern(size=dim)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(stick_propn_mean, stick_propn_info, centroids, gamma,
                        prior_params_dict,
                        gh_loc, gh_weights):
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info,
                                            alpha, gh_loc, gh_weights)

    # wishart prior
    df = prior_params_dict['prior_gamma_df']
    V_inv = prior_params_dict['prior_gamma_inv_scale']
    e_gamma_prior = modeling_lib.get_e_log_wishart_prior(gamma, df, V_inv)

    # centroid prior
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_info = prior_params_dict['prior_centroid_info']
    e_centroid_prior = \
        modeling_lib.get_e_centroid_prior(centroids, prior_mean, prior_info)
    
    return np.squeeze(e_gamma_prior + e_centroid_prior + dp_prior)

##########################
# Entropy
##########################
def get_entropy(stick_propn_mean, stick_propn_info, e_z, gh_loc, gh_weights,
                    use_logitnormal_sticks = True):
    # get entropy term

    z_entropy = modeling_lib.multinom_entropy(e_z)
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(stick_propn_mean, stick_propn_info,
                                gh_loc, gh_weights)

    return z_entropy + stick_entropy

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, centroids, gamma):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster

    dim = np.shape(y)[1]

    assert np.shape(y)[1] == np.shape(centroids)[0]
    assert np.shape(gamma)[0] == np.shape(centroids)[1]
    assert np.shape(gamma)[1] == np.shape(centroids)[0]

    data2_term = np.einsum('ni, kij, nj -> nk', y, gamma, y)
    cross_term = np.einsum('ni, kij, jk -> nk', y, gamma, centroids)
    centroid2_term = np.einsum('ik, kij, jk -> k', centroids, gamma, centroids)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(centroid2_term, axis = 0)

    return - 0.5 * squared_term + 0.5 * np.linalg.slogdet(gamma)[1][None, :]

##########################
# Optimization over e_z
##########################

def get_z_nat_params(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                        gh_loc, gh_weights,
                        use_bnp_prior = True):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y, centroids, gamma)

    # get weight term
    if use_bnp_prior:
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities(
                            stick_propn_mean, stick_propn_info,
                            gh_loc, gh_weights)
    else:
        e_log_cluster_probs = 0.

    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param, loglik_obs_by_nk

def get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                    gh_loc, gh_weights,
                    use_bnp_prior = True):

    z_nat_param, loglik_obs_by_nk= \
        get_z_nat_params(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                                    gh_loc, gh_weights,
                                    use_bnp_prior)

    log_const = sp.misc.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - log_const[:, None])

    return e_z, loglik_obs_by_nk

def get_optimal_z_from_vb_params_dict(y, vb_params_dict, gh_loc, gh_weights,
                                        use_bnp_prior = True):

    """
    Returns the optimal cluster belonging probabilities, given the
    variational parameters.

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    e_z : ndarray
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k

    """

    # get global vb parameters
    stick_propn_mean = vb_params_dict['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_propn_info']
    centroids = vb_params_dict['centroids']
    gamma = vb_params_dict['gamma']

    # compute optimal e_z from vb global parameters
    e_z, _ = get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                        gh_loc, gh_weights,
                        use_bnp_prior = True)

    return e_z


def get_kl(y, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights,
                    e_z = None,
                    data_weights = None,
                    use_bnp_prior = True):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    prior_params_dict : dictionary
        Dictionary of prior parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    e_z : ndarray (optional)
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k.
        If ``None``, we set the optimal z.
    data_weights : ndarray of shape (number of observations) x 1 (optional)
        Weights for each datapoint in y.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    kl : float
        The negative elbo.
    """
    # get vb parameters
    stick_propn_mean = vb_params_dict['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_propn_info']
    centroids = vb_params_dict['centroids']
    gamma = vb_params_dict['gamma']

    # get optimal cluster belongings
    e_z_opt, loglik_obs_by_nk = \
            get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                            gh_loc, gh_weights)
    if e_z is None:
        e_z = e_z_opt


    # weight data if necessary, and get likelihood of y
    if data_weights is not None:
        assert np.shape(data_weights)[0] == n_obs, \
                    'data weights need to be n_obs by 1'
        assert np.shape(data_weights)[1] == 1, \
                    'data weights need to be n_obs by 1'
        e_loglik_obs = np.sum(data_weights * e_z * loglik_obs_by_nk)
    else:
        e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

    # likelihood of z
    if use_bnp_prior:
        e_loglik_ind = modeling_lib.loglik_ind(stick_propn_mean, stick_propn_info, e_z,
                            gh_loc, gh_weights)
    else:
        e_loglik_ind = 0.

    e_loglik = e_loglik_ind + e_loglik_obs

    if not np.isfinite(e_loglik):
        print('gamma', vb_params_dict['gamma'].get())
        print('det gamma', np.linalg.slogdet(
            vb_params_dict['gamma'])[1])
        print('cluster weights', np.sum(e_z, axis = 0))

    assert(np.isfinite(e_loglik))

    # entropy term
    entropy = np.squeeze(get_entropy(stick_propn_mean, stick_propn_info, e_z,
                                        gh_loc, gh_weights))
    assert(np.isfinite(entropy))

    # prior term
    e_log_prior = get_e_log_prior(stick_propn_mean, stick_propn_info, centroids, gamma,
                            prior_params_dict,
                            gh_loc, gh_weights)

    assert(np.isfinite(e_log_prior))

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo



########################
# Posterior quantities of interest
#######################

def get_e_num_pred_clusters_from_vb_free_params(vb_params_paragami,
                                                    vb_params_free,
                                                    n_obs,
                                                    threshold = 0,
                                                    n_samples = 100000):
    # get posterior predicted number of clusters

    vb_params_dict = \
        vb_params_paragami.fold(vb_params_free, free = True)

    mu = vb_params_dict['stick_propn_mean']
    sigma = 1 / np.sqrt(vb_params_dict['stick_propn_info'])

    return cluster_lib.get_e_number_clusters_from_logit_sticks(mu, sigma,
                                                        n_obs,
                                                        threshold = threshold,
                                                        n_samples = n_samples)


# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_free_par(y, vb_params_paragami, vb_params_free,
                                        gh_loc, gh_weights,
                                        threshold = 0,
                                        n_samples = 100000):

    vb_params_dict = \
        vb_params_paragami.fold(vb_params_free, free = True)

    e_z  = get_optimal_z_from_vb_params_dict(y, vb_params_dict, gh_loc, gh_weights,
                                            use_bnp_prior = True)

    return cluster_lib.get_e_num_large_clusters_from_ez(e_z,
                                        threshold = threshold,
                                        n_samples = 100000)
