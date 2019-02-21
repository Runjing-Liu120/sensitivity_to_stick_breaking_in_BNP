import autograd
import autograd.numpy as np
import autograd.scipy as sp

import paragami

import sys
sys.path.insert(0, '../../BNP_modeling/')
import modeling_lib
import cluster_quantities_lib

import LinearResponseVariationalBayes.ExponentialFamilies as ef

from sklearn.decomposition import NMF

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(n_obs, n_loci, k_approx):
    """
    Returns a paragami patterned dictionary
    that stores the variational parameters.

    Parameters
    ----------

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # variational beta parameters for population allele frequencies
    vb_params_paragami['pop_freq_beta_params'] = \
        paragami.NumericArrayPattern(shape=(n_loci, k_approx, 2), lb = 0.0)

    # BNP sticks
    # variational distribution for each stick is logitnormal
    vb_params_paragami['ind_mix_stick_propn_mean'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,))
    vb_params_paragami['ind_mix_stick_propn_info'] = \
        paragami.NumericArrayPattern(shape = (n_obs, k_approx - 1,), lb = 1e-4)

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami


##########################
# Set up prior parameters
##########################
def get_default_prior_params():
    """
    Returns a paragami patterned dictionary
    that stores the prior parameters.

    Returns
    -------
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    prior_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the prior parameters.

    """

    prior_params_dict = dict()
    prior_params_paragami = paragami.PatternDict()

    # DP prior parameter for the individual admixtures
    prior_params_dict['dp_prior_alpha'] = np.array([3.0])
    prior_params_paragami['dp_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the allele frequencies
    # beta distribution parameters
    prior_params_dict['allele_prior_alpha'] = np.array([1.])
    prior_params_dict['allele_prior_beta'] = np.array([1.])
    prior_params_paragami['allele_prior_alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)
    prior_params_paragami['allele_prior_beta'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                        e_log_p, e_log_1mp,
                        dp_prior_alpha, allele_prior_alpha,
                        allele_prior_beta,
                        gh_loc, gh_weights):
    # get expected prior term

    # dp prior on individual mixtures
    ind_mix_dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(ind_mix_stick_propn_mean,
                                            ind_mix_stick_propn_info,
                                            dp_prior_alpha, gh_loc, gh_weights)

    # allele frequency prior
    allele_freq_beta_prior = np.sum((allele_prior_alpha - 1) * e_log_p + \
                                    (allele_prior_beta - 1) * e_log_1mp)

    return ind_mix_dp_prior + allele_freq_beta_prior

##########################
# Entropy
##########################
def get_entropy(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                    pop_freq_beta_params,
                    e_z, gh_loc, gh_weights):
    # get entropy term

    # entropy on population belongings

    z_entropy = -(np.log(e_z + 1e-12) * e_z).sum()

    # entropy of individual admixtures
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(
                                ind_mix_stick_propn_mean,
                                ind_mix_stick_propn_info,
                                gh_loc, gh_weights)

    # beta entropy term
    lk = pop_freq_beta_params.shape[0] * pop_freq_beta_params.shape[1]
    beta_entropy = ef.beta_entropy(tau = pop_freq_beta_params.reshape((lk, 2)))

    return z_entropy + stick_entropy + beta_entropy

##########################
# Likelihood term
##########################
def get_loglik_gene_nlk(g_obs, e_log_p, e_log_1mp):

    genom_loglik_nlk_a = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0], e_log_1mp) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 1] + g_obs[:, :, 2], e_log_p)

    genom_loglik_nlk_b = \
        np.einsum('nl, lk -> nlk', g_obs[:, :, 0] + g_obs[:, :, 1], e_log_1mp) + \
            np.einsum('nl, lk -> nlk', g_obs[:, :, 2], e_log_p)

    return np.stack((genom_loglik_nlk_a, genom_loglik_nlk_b), axis = -1)

##########################
# Optimization over e_z
##########################
def get_loglik_cond_z(g_obs, e_log_p, e_log_1mp,
                        ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                        gh_loc, gh_weights,
                        true_ind_admix_propn = None):

    # get likelihood of genes
    loglik_gene_nlk = get_loglik_gene_nlk(g_obs, e_log_p, e_log_1mp)

    # log likelihood of population belongings
    n = ind_mix_stick_propn_mean.shape[0]
    k = ind_mix_stick_propn_mean.shape[1] + 1

    if true_ind_admix_propn is None:
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities(
                            ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                            gh_loc, gh_weights).reshape(n, 1, k, 1)
    else:
        e_log_cluster_probs = np.log(true_ind_admix_propn).reshape(n, 1, k, 1)

    # loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_gene_nlk + e_log_cluster_probs

    return loglik_cond_z

def get_z_opt_from_loglik_cond_z(loglik_cond_z):
    # 2nd axis dimension is k
    # recall that loglik_obs_by_nlk2 is n_obs x n_loci x k_approx x 2
    loglik_cond_z = loglik_cond_z - np.max(loglik_cond_z, axis = 2, keepdims = True)

    log_const = sp.misc.logsumexp(loglik_cond_z, axis = 2, keepdims = True)

    return np.exp(loglik_cond_z - log_const)

def get_kl(g_obs, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights,
                    e_z = None,
                    data_weights = None,
                    true_pop_allele_freq = None,
                    true_ind_admix_propn = None):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    g_obs : ndarray
        The array of one-hot encoded genotypes, of shape (n_obs, n_loci, 3)
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
        parameters, stored in an array whose (n, l, k, i)th entry is the probability
        of the nth datapoint at locus l and chromosome i belonging to cluster k.
        If ``None``, we set the optimal z.
    data_weights : ndarray of shape (number of observations) x 1 (optional)
        Weights for each datapoint in g_obs.

    Returns
    -------
    kl : float
        The negative elbo.
    """
    # get prior parameters
    dp_prior_alpha = prior_params_dict['dp_prior_alpha']
    allele_prior_alpha = prior_params_dict['allele_prior_alpha']
    allele_prior_beta = prior_params_dict['allele_prior_beta']

    # get vb parameters
    ind_mix_stick_propn_mean = vb_params_dict['ind_mix_stick_propn_mean']
    ind_mix_stick_propn_info = vb_params_dict['ind_mix_stick_propn_info']
    pop_freq_beta_params = vb_params_dict['pop_freq_beta_params']

    # expected log beta and expected log(1 - beta)
    if true_pop_allele_freq is None:
        e_log_p, e_log_1mp = modeling_lib.get_e_log_beta(pop_freq_beta_params)
    else:
        e_log_p = np.log(true_pop_allele_freq + 1e-8)
        e_log_1mp = np.log(1 - true_pop_allele_freq + 1e-8)

    # get optimal cluster belongings
    loglik_cond_z = \
            get_loglik_cond_z(g_obs, e_log_p, e_log_1mp,
                            ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                            gh_loc, gh_weights,
                            true_ind_admix_propn = true_ind_admix_propn)

    e_z_opt = get_z_opt_from_loglik_cond_z(loglik_cond_z)

    if e_z is None:
        e_z = e_z_opt

    # weight data if necessary, and get likelihood of y
    if data_weights is not None:
        raise NotImplementedError()
    else:
        e_loglik = np.sum(e_z * loglik_cond_z)

    assert(np.isfinite(e_loglik))

    # entropy term
    entropy = get_entropy(ind_mix_stick_propn_mean,
                                        ind_mix_stick_propn_info,
                                        pop_freq_beta_params,
                                        e_z, gh_loc, gh_weights).squeeze()
    assert(np.isfinite(entropy))

    # prior term
    e_log_prior = get_e_log_prior(ind_mix_stick_propn_mean, ind_mix_stick_propn_info,
                            e_log_p, e_log_1mp,
                            dp_prior_alpha, allele_prior_alpha,
                            allele_prior_beta,
                            gh_loc, gh_weights).squeeze()

    assert(np.isfinite(e_log_prior))

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo


###############
# functions for initializing
def cluster_and_get_init(g_obs, k):
    # g_obs should be n_obs x n_loci x 3,
    # a one-hot encoding of genotypes
    assert len(g_obs.shape) == 3

    # convert one-hot encoding to probability of A genotype, {0, 0.5, 1}
    x = g_obs.argmax(axis = 2) / 2

    # run NMF
    model = NMF(n_components=k, init='random')
    init_ind_admix_propn_unscaled = model.fit_transform(x)
    init_pop_allele_freq_unscaled = model.components_.T

    # divide by largest allele frequency, so all numbers between 0 and 1
    denom_pop_allele_freq = np.max(init_pop_allele_freq_unscaled)
    init_pop_allele_freq = init_pop_allele_freq_unscaled / \
                                denom_pop_allele_freq

    # normalize rows
    denom_ind_admix_propn = \
        init_ind_admix_propn_unscaled.sum(axis = 1, keepdims = True)
    init_ind_admix_propn = \
        init_ind_admix_propn_unscaled / denom_ind_admix_propn
    # clip again and renormalize
    init_ind_admix_propn = init_ind_admix_propn.clip(0.05, 0.95)
    init_ind_admix_propn = init_ind_admix_propn / \
                            init_ind_admix_propn.sum(axis = 1, keepdims = True)

    return init_ind_admix_propn, init_pop_allele_freq.clip(0.05, 0.95)

def set_init_vb_params(g_obs, k_approx, vb_params_dict):
    # get initial admixtures, and population frequencies
    init_ind_admix_propn, init_pop_allele_freq = \
            cluster_and_get_init(g_obs, k_approx)

    # set bnp parameters for individual admixture
    # set mean to be logit(stick_breaking_propn), info to be 1
    stick_break_propn = \
        cluster_quantities_lib.get_stick_break_propns_from_mixture_weights(init_ind_admix_propn)
    ind_mix_stick_propn_mean = np.log(stick_break_propn) - np.log(1 - stick_break_propn)
    ind_mix_stick_propn_info = np.ones(stick_break_propn.shape)

    # set beta paramters for population paramters
    # set beta = 1, alpha to have the correct mean
    pop_freq_beta_params1 = init_pop_allele_freq / (1 - init_pop_allele_freq)
    pop_freq_beta_params2 = np.ones(init_pop_allele_freq.shape)
    pop_freq_beta_params = np.concatenate((pop_freq_beta_params1[:, :, None],
                                       pop_freq_beta_params2[:, :, None]), axis = 2)

    vb_params_dict['ind_mix_stick_propn_mean'] = ind_mix_stick_propn_mean
    vb_params_dict['ind_mix_stick_propn_info'] = ind_mix_stick_propn_info
    vb_params_dict['pop_freq_beta_params'] = pop_freq_beta_params

    return vb_params_dict
