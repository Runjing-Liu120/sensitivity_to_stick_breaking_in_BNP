import autograd.numpy as np
import autograd.scipy as sp

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef


def _cumprod_through_log(x, axis = None):
    # a replacement for np.cumprod since
    # Autograd doesn't work with the original cumprod.
    return np.exp(np.cumsum(np.log(x), axis = axis))

def get_mixture_weights_from_stick_break_propns(stick_break_propns):
    """
    Computes stick lengths (i.e. mixture weights) from stick breaking
    proportions.

    Parameters
    ----------
    stick_break_propns : ndarray
        Array of stick breaking proportions, with sticks along last dimension

    Returns
    -------
    mixture_weights : ndarray
        An array  the same size as stick_break_propns,
        with the mixture weights computed for each row of
        stick breaking proportions.

    """

    # if input is a vector, make it a 1 x k_approx array
    if len(np.shape(stick_break_propns)) == 1:
        stick_break_propns = np.array([stick_break_propns])

    # number of components
    k_approx = np.shape(stick_break_propns)[-1]
    # number of mixtures
    ones_shape = stick_break_propns.shape[0:-1] + (1,)

    stick_break_propns_1m = 1 - stick_break_propns
    stick_remain = np.concatenate((np.ones(ones_shape),
                        _cumprod_through_log(stick_break_propns_1m, axis = -1)), axis = -1)
    stick_add = np.concatenate((stick_break_propns,
                                np.ones(ones_shape)), axis = -1)

    mixture_weights = (stick_remain * stick_add).squeeze()

    return mixture_weights

def get_e_cluster_probabilities(stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights):
    e_cluster_probs = \
        ef.get_e_logitnormal(stick_propn_mean, stick_propn_info, gh_loc, gh_weights)

    return get_mixture_weights_from_stick_break_propns(e_cluster_probs)

def get_stick_break_propns_from_mixture_weights(mixture_weights):
    """
    Computes stick breaking proportions from the mixture weights. Used
    for initializing vb parameters

    Parameters
    ----------
    stick_break_propns : ndarray
        Array of stick breaking proportions, with sticks along last dimension

    Returns
    -------
    stick_break_propns : ndarray
        Array of stick breaking proportions, with sticks along last dimension
    """

    # TODO: implement for more general array shapes
    if not len(np.shape(mixture_weights)) == 2:
        raise NotImplementedError

    n_obs = mixture_weights.shape[0]
    k_approx = mixture_weights.shape[1]

    stick_break_propn = np.zeros((n_obs, k_approx - 1))
    stick_remain = np.ones(n_obs)

    for i in range(k_approx - 1):
        stick_break_propn[:, i] = mixture_weights[:, i] / stick_remain
        stick_remain *= (1 - stick_break_propn[:, i])

    return stick_break_propn


def get_e_number_clusters_from_logit_sticks(stick_propn_mean, stick_propn_info,
                                            n_obs, threshold = 0,
                                            n_samples = None,
                                            unv_norm_samples = None):
    """
    Computes the expected number of clusters with at least t observations
    from logitnormal stick-breaking parameters,
    ``stick_propn_mean`` and ``stick_propn_info``,
    using Monte Carlo.

    Parameters
    ----------
    stick_propn_mean : ndarray
        Mean parameters for the logit of the
        stick-breaking proportions, of shape ...  x k_approx
    stick_propn_info : ndarray
        parameters for the logit of the
        stick-breaking proportions, of shape ...  x k_approx
    threshold : int
        Miniumum number of observations for a cluster to be counted.
    n_obs : int
        Number of observations in a dataset
    n_samples : int
        Number of Monte Carlo samples used to compute the expected
        number of clusters.
    unv_norm_samples : ndarray, optional
        The user may pass in a precomputed array of uniform random variables
        on which the reparameterization trick is applied to compute the
        expected number of clusters.

    Returns
    -------
    float
        The expected number of clusters with at least ``threshold`` observations
        in a dataset of size n_obs
    """

    assert stick_propn_mean.shape == stick_propn_info.shape
    if len(stick_propn_mean.shape) == 1:
        stick_propn_mean = stick_propn_mean[None, :]
        stick_propn_info = stick_propn_info[None, :]

    assert (n_samples is not None) or (unv_norm_samples is not None), \
        'both n_samples and unv_norm_samples cannot be None'

    unif_samples_shape = (n_samples, ) + stick_propn_mean.shape
    if unv_norm_samples is None:
        unv_norm_samples = np.random.normal(0, 1, size = unif_samples_shape)
    if n_samples is None:
        assert unv_norm_samples.shape == unif_samples_shape

    # sample sticks proportions from logitnormal
    # this is n_samples x ... x (k_approx - 1)
    stick_propn_samples = sp.special.expit(unv_norm_samples * \
                            1 / np.sqrt(stick_propn_info) + \
                                        stick_propn_mean)

    # get posterior weights
    # this is n_samples x ... x (k_approx - 1)
    weight_samples = \
        get_mixture_weights_from_stick_break_propns(stick_propn_samples)

    # compute expected number of clusters with at least threshold datapoints
    subtr_weight = (1 - weight_samples)**(n_obs)
    assert isinstance(threshold, int)
    for i in range(1, threshold):
        subtr_weight += \
            osp.special.comb(n_obs, i) * \
            weight_samples**i * (1 - weight_samples)**(n_obs - i)

    return np.mean(np.sum(1 - subtr_weight, axis = -1), axis = 0).squeeze()


def get_e_num_clusters_from_ez(e_z):
    """
    Analytically computes the expected number of clusters from cluster
    belongings e_z.

    Parameters
    ----------
    e_z : ndarray
        Array whose (n, k)th entry is the probability of the nth
        datapoint belonging to cluster k.

    Returns
    -------
    float
        The expected number of clusters in the dataset.

    """

    k = np.shape(e_z)[1]
    return k - np.sum(np.prod(1 - e_z, axis = 0))

def _get_clusters_from_ez_and_unif_samples(e_z_cumsum, unif_samples):
    # returns a n_obs x n_samples binary  matrix encoding the cluster belonging
    # of the nth observation in nth sample

    n_obs = e_z_cumsum.shape[0]

    # unif_sample should be a matrix of shape n_obs x n_samples
    # assert len(unif_samples.shape) == 2
    # assert unif_samples.shape[0] == n_obs

    # get which cluster the sample belongs to
    # z_sample = (e_z_cumsum[:, :, None] > unif_samples[:, None, :]).argmax(-2)
    z_sample = (np.expand_dims(e_z_cumsum, axis = -1) >
                np.expand_dims(unif_samples, axis = -2)).argmax(-2)

    return z_sample


def get_e_num_large_clusters_from_ez(e_z,
                                    threshold = 0,
                                    n_samples = None,
                                    unif_samples = None):
    """
    Computes the expected number of clusters with at least t
    observations from cluster belongings e_z.

    Parameters
    ----------
    e_z : ndarray
        Array whose (..., k)th entry is the probability of the ...th
        datapoint belonging to cluster k
    n_obs : int
        Number of observations in a dataset.
    n_samples : int
        Number of Monte Carlo samples used to compute the expected
        number of clusters.
    unv_norm_samples : ndarray, optional
        The user may pass in a precomputed array of uniform random variables
        on which the reparameterization trick is applied to compute the
        expected number of clusters.

    Returns
    -------
    float
        The expected number of clusters with at least ``threshold`` observations
        in a dataset the same size as e_z

    """

    obs_shape = e_z.shape[0:-1]
    n_clusters = e_z.shape[-1]

    # draw uniform samples
    if unif_samples is None:
        assert n_samples is not None
        unif_samples = np.random.random(obs_shape + (n_samples,))

    else:
        assert unif_samples is not None
        assert unif_samples.shape[0:-1] == obs_shape

    n_samples = unif_samples.shape[-1]
    e_z_cumsum = np.cumsum(e_z, axis = -1)

    num_heavy_clusters_vec = np.zeros(n_samples)

    # z_sample is a obs_shape x n_samples matrix of cluster belongings
    z_sample = _get_clusters_from_ez_and_unif_samples(e_z_cumsum, unif_samples)

    for i in range(n_clusters):
        # get number of clusters with at least enough points above the threshold
        num_heavy_clusters_vec += \
            (z_sample == i).reshape(-1, n_samples).sum(axis = 0) > threshold

    return np.mean(num_heavy_clusters_vec), np.var(num_heavy_clusters_vec)
