import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.ExponentialFamilies as ef

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp
################
# define entropies

def multinom_entropy(e_z):
    # returns the entropy of the cluster belongings
    return -1 * np.sum(e_z * np.log(e_z + 1e-8))

def get_stick_breaking_entropy(stick_propn_mean, stick_propn_info,
                                gh_loc, gh_weights):
    # return the entropy of logitnormal distriibution on the sticks whose
    # logit has mean stick_propn_mean and information stick_propn_info
    # Integration is done on the real line with respect to the Lesbegue measure

    # integration is done numerical with Gauss Hermite quadrature.
    # gh_loc and gh_weights specifiy the location and weights of the
    # quadrature points

    # we seek E[log q(V)], where q is the density of a logit-normal, and
    # V ~ logit-normal. Let W := logit(V), so W ~ Normal. Hence,
    # E[log q(W)]; we can then decompose log q(x) into the terms of a normal
    # distribution and the jacobian term. The expectation of the normal term
    # evaluates to the normal entropy, and we add the jacobian term to it.
    # The jacobian term is 1/(x(1-x)), so we simply add -EV - E(1-V) to the normal
    # entropy.

    assert np.all(gh_weights > 0)

    assert len(stick_propn_mean) == len(stick_propn_info)
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv =\
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return np.sum(ef.univariate_normal_entropy(stick_propn_info)) + \
                    np.sum(e_log_v + e_log_1mv)

################
# define priors
def get_e_centroid_prior(centroids, prior_mean, prior_info):
    # expected log prior for cluster centroids
    # Note that the variational distribution for the centroid is a dirac
    # delta function

    assert prior_info > 0

    beta_base_prior = ef.uvn_prior(prior_mean = prior_mean,
                                    prior_info = prior_info,
                                    e_obs = centroids.flatten(),
                                    var_obs = np.array([0.]))

    return np.sum(beta_base_prior)

def get_e_log_wishart_prior(gamma, df, V_inv):
    # expected log prior for cluster info matrices gamma

    dim = V_inv.shape[0]

    assert np.shape(gamma)[1] == dim

    tr_V_inv_gamma = np.einsum('ij, kji -> k', V_inv, gamma)

    s, logdet = np.linalg.slogdet(gamma)
    assert np.all(s > 0), 'some gammas are not PSD'

    return np.sum((df - dim - 1) / 2 * logdet - 0.5 * tr_V_inv_gamma)

# Get a vector of expected functions of the logit sticks.
# You can use this to define proportional functional perturbations to the
# logit stick distributions.
# The function func should take arguments in the logit stick space, i.e.
# logit_stick = log(stick / (1 - stick)).
def get_e_func_logit_stick_vec(vb_params_dict, gh_loc, gh_weights, func):
    stick_propn_mean = vb_params_dict['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_propn_info']

    # print('DEBUG: 0th lognorm mean: ', stick_propn_mean[0])
    e_phi = np.array([
        ef.get_e_fun_normal(
            stick_propn_mean[k], stick_propn_info[k], \
            gh_loc, gh_weights, func)
        for k in range(len(stick_propn_mean))
    ])

    return e_phi


def get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info, alpha,
                                gh_loc, gh_weights):
    # expected log prior for the stick breaking proportions under the
    # logitnormal variational distribution

    # integration is done numerical with Gauss Hermite quadrature.
    # gh_loc and gh_weights specifiy the location and weights of the
    # quadrature points

    assert np.all(gh_weights > 0)

    assert len(stick_propn_mean) == len(stick_propn_info)
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    return (alpha - 1) * np.sum(e_log_1mv)


##############
# likelihoods
def get_e_log_cluster_probabilities(stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights):

    # the expected log mixture weights

    assert np.all(gh_weights > 0)

    assert len(stick_propn_mean) == len(stick_propn_info)
    assert np.all(stick_propn_info) > 0

    e_log_v, e_log_1mv = \
        ef.get_e_log_logitnormal(
            lognorm_means = stick_propn_mean,
            lognorm_infos = stick_propn_info,
            gh_loc = gh_loc,
            gh_weights = gh_weights)

    e_log_stick_remain = np.concatenate([np.array([0.]), np.cumsum(e_log_1mv)])
    e_log_new_stick = np.concatenate((e_log_v, np.array([0])))

    return e_log_stick_remain + e_log_new_stick


def loglik_ind(stick_propn_mean, stick_propn_info, e_z, gh_loc, gh_weights):

    # likelihood of cluster belongings e_z

    assert np.all(gh_weights > 0)

    assert len(stick_propn_mean) == len(stick_propn_info)
    assert np.all(stick_propn_info) > 0


    # expected log likelihood of all indicators for all n observations
    e_log_cluster_probs = \
        get_e_log_cluster_probabilities(stick_propn_mean, stick_propn_info,
                                        gh_loc, gh_weights)

    return np.sum(e_z * e_log_cluster_probs)
