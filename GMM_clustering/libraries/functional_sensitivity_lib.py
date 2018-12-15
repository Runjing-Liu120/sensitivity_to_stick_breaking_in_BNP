# some extra functions to do functional sensitivity

import autograd.numpy as np
import autograd.scipy as sp

import gmm_clustering_lib as gmm_lib
import modeling_lib as model_lib

import scipy as osp

from copy import deepcopy

class PriorPerturbation(object):
    def __init__(self, vb_params_dict, alpha0,
                        log_phi, gh_loc,
                        gh_weights, epsilon=1.0,
                        logit_v_ub = 4,
                        logit_v_lb = -4,
                        quad_maxiter = 50):

        self.logit_v_lb = logit_v_lb
        self.logit_v_ub = logit_v_ub

        self.gh_loc = gh_loc
        self.gh_weights = gh_weights

        self.quad_maxiter = quad_maxiter

        self.gustafson_style = False

        self.vb_params_dict = vb_params_dict
        self.alpha0 = alpha0

        self.epsilon_param = epsilon

        self.set_log_phi(log_phi)

    #################
    # Functions that are used for graphing and the influence function.

    # The log variational density of stick k at logit_v
    # in the logit_stick space.
    def get_log_q_logit_stick(self, logit_v, k):
        mean = self.vb_params_dict['stick_propn_mean']
        info = self.vb_params_dict['stick_propn_info']
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    # Return a vector of log variational densities for all sticks at logit_v
    # in the logit stick space.
    def get_log_q_logit_all_sticks(self, logit_v):
        mean = self.vb_params_dict['stick_propn_mean']
        info = self.vb_params_dict['stick_propn_info']
        return -0.5 * (info * (logit_v - mean) ** 2 - np.log(info))

    def get_log_p0(self, v):
        alpha = self.alpha0
        return (alpha - 1) * np.log1p(-v) - self.log_norm_p0

    def get_log_p0_logit(self, logit_v):
        alpha = self.alpha0
        return \
            - alpha * logit_v - (alpha + 1) * np.log1p(np.exp(-logit_v)) - \
            self.log_norm_p0_logit

    def get_log_pc(self, v):
        logit_v = np.log(v) - np.log(1 - v)
        epsilon = self.epsilon_param
        if np.abs(epsilon) < 1e-8:
            return self.get_log_p0(v)

        if self.gustafson_style:
            log_epsilon = np.log(epsilon)
            return \
                self.get_log_p0(v) + \
                self.log_phi(logit_v) + \
                log_epsilon - \
                self.log_norm_pc
        else:
            # assert epsilon <= 1
            return \
                self.get_log_p0(v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc

    def get_log_pc_logit(self, logit_v):
        epsilon = self.epsilon_param
        if np.abs(epsilon) < 1e-8:
            return self.get_log_p0_logit(logit_v)

        if self.gustafson_style:
            log_epsilon = np.log(epsilon)
            return \
                self.get_log_p0_logit(logit_v) + \
                self.log_phi(logit_v) + \
                log_epsilon - \
                self.log_norm_pc_logit
        else:
            # assert epsilon <= 1
            return \
                self.get_log_p0_logit(logit_v) + \
                epsilon * self.log_phi(logit_v) - \
                self.log_norm_pc_logit

    ###################################
    # Setting functions for initialization

    def set_epsilon(self, epsilon):
        self.epsilon_param = epsilon
        self.set_log_phi(self.log_phi)

    def set_log_phi(self, log_phi):
        # Set attributes derived from phi and epsilon

        # Initial values for the log normalzing constants which will be set below.
        self.log_norm_p0 = 0
        self.log_norm_pc = 0
        self.log_norm_p0_logit = 0
        self.log_norm_pc_logit = 0

        self.log_phi = log_phi

        norm_p0, _ = osp.integrate.quadrature(
            lambda v: np.exp(self.get_log_p0(v)), 0, 1, maxiter = self.quad_maxiter)
        assert norm_p0 > 0
        self.log_norm_p0 = np.log(norm_p0)

        norm_pc, _ = osp.integrate.quadrature(
            lambda v: np.exp(self.get_log_pc(v)),
            0, 1, maxiter = self.quad_maxiter)
        assert norm_pc > 0
        self.log_norm_pc = np.log(norm_pc)

        norm_p0_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_p0_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub, maxiter = self.quad_maxiter)
        assert norm_p0_logit > 0
        self.log_norm_p0_logit = np.log(norm_p0_logit)

        norm_pc_logit, _ = osp.integrate.quadrature(
            lambda logit_v: np.exp(self.get_log_pc_logit(logit_v)),
            self.logit_v_lb, self.logit_v_ub, maxiter = self.quad_maxiter)
        assert norm_pc_logit > 0
        self.log_norm_pc_logit = np.log(norm_pc_logit)


def get_e_log_perturbation(log_phi, vb_params_dict, epsilon_param_dict,
                           gh_loc, gh_weights, sum_vector=True):

    """
    Computes the expected log multiplicative perturbation

    Parameters
    ----------
    log_phi : Callable function
        The log of the multiplicative perturbation in logit space
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters
    epsilon_param_dict : dictionary
        Dictionary with key 'epsilon' specififying the multiplicative perturbation
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    sum_vector : boolean
        whether to sum the expectation over the k sticks

    Returns
    -------
    float
        The expected log perturbation under the variational distribution

    """

    perturbation_fun = \
        lambda logit_v: log_phi(logit_v) * epsilon_param_dict['epsilon']

    e_perturbation_vec = model_lib.get_e_func_logit_stick_vec(
        vb_params_dict, gh_loc, gh_weights, perturbation_fun)

    if sum_vector:
        return -1 * np.sum(e_perturbation_vec)
    else:
        return -1 * e_perturbation_vec

def get_perturbed_kl(y, vb_params_dict, epsilon_param_dict, log_phi,
                     prior_params_dict, gh_loc, gh_weights):

    """
    Computes KL divergence after perturbing by log_phi

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters
    epsilon_param_dict : dictionary
        Dictionary with key 'epsilon' specififying the multiplicative perturbation
    log_phi : Callable function
        The log of the multiplicative perturbation in logit space
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    sum_vector : boolean
        whether to sum the expectation over the k sticks

    Returns
    -------
    float
        The KL divergence after perturbing by log_phi

    """

    e_log_pert = get_e_log_perturbation(log_phi, vb_params_dict,
                            epsilon_param_dict,
                            gh_loc, gh_weights, sum_vector=True)

    return gmm_lib.get_kl(y, vb_params_dict,
                            prior_params_dict, gh_loc, gh_weights) + e_log_pert
