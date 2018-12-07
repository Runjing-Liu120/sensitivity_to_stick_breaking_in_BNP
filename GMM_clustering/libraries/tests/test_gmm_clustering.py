#!/usr/bin/env python3

import sys
sys.path.insert(0, '../libraries')

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

import gmm_clustering_lib as gmm_lib
import simulation_lib

from numpy.polynomial.hermite import hermgauss

import unittest

import numpy.testing as testing


class TestOptimalEz(unittest.TestCase):
    def test_set_z_nat_params(self):
        # Test whether our function to set the z natural parameters as a
        # function of the global parameters actually returns the optimal z.
        # To do so, we check that the gradient wrt to z at the optimum is 0

        np.random.seed(456456)

        # simulate data from gmm mixture
        n_obs = 1000
        dim = 2
        true_k = 5
        y = simulation_lib.simulate_data(n_obs, dim, true_k, separation=0.2)[0]

        # setup vb_parameters
        _, vb_params_paragami = \
            gmm_lib.get_vb_params_paragami_object(dim, true_k)
        vb_params_dict = vb_params_paragami.random()

        # prior parameters
        prior_params_dict, prior_params_paragami = \
            gmm_lib.get_default_prior_params(dim)

        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        # function that returns kl as a function of the z natural parameters
        def get_kl_from_z_nat_param(y, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights, z_nat_param):

            log_const = sp.misc.logsumexp(z_nat_param, axis=1)
            e_z = np.exp(z_nat_param - log_const[:, None])

            return gmm_lib.get_kl(y, vb_params_dict, prior_params_dict,
                            gh_loc, gh_weights,
                            e_z = e_z)

        # get z natural parameters
        stick_propn_mean = vb_params_dict['stick_propn_mean']
        stick_propn_info = vb_params_dict['stick_propn_info']
        centroids = vb_params_dict['centroids']
        gamma = vb_params_dict['gamma']

        z_nat_param, _ = \
            gmm_lib.get_z_nat_params(y, stick_propn_mean,
                                        stick_propn_info, centroids, gamma,
                                        gh_loc, gh_weights)

        # compute gradient at z_nat_param
        kl_z_nat_param_grad = autograd.grad(get_kl_from_z_nat_param, argnum = 5)

        grad_at_opt = kl_z_nat_param_grad(y, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights, z_nat_param)

        # assert gradient is 0
        testing.assert_allclose(grad_at_opt, 0, rtol = 0.0, atol = 1e-8)

# TODO: this test NOT finished yet
# class TestOptimization(unittest.TestCase):
#     def test_optimization_on_simple_mixture(self):
#
#         # simulate data from gmm mixture
#         n_obs = 1000
#         dim = 2
#         true_k = 5
#         y, true_z, true_components, true_centroids, true_covs, true_probs = \
#             simulation_lib.simulate_data(n_obs, dim, true_k, separation=0.2)
#
#         # setup vb_parameters
#         _, vb_params_paragami = \
#             gmm_lib.get_vb_params_paragami_object(dim, true_k)
#         vb_params_dict = vb_params_paragami.random()
#         random_vb_free_init = \
#             vb_params_paragami.flatten(vb_params_dict, free = True)
#
#         # prior parameters
#         prior_params_dict, prior_params_paragami = \
#             gmm_lib.get_default_prior_params(dim)
#
#         gh_deg = 8
#         gh_loc, gh_weights = hermgauss(gh_deg)
#
#
#         vb_opt = gmm_lib.optimize_full(y, vb_params_paragami, prior_params_dict,
#                     random_vb_free_init, gh_loc, gh_weights,
#                     bfgs_max_iter = 10, netwon_max_iter = 50,
#                     max_precondition_iter = 10,
#                     gtol=1e-8, ftol=1e-8, xtol=1e-8)
#
#         vb_opt_dict = vb_params_paragami.fold(vb_opt, free = True)
#
#         print(vb_opt_dict['centroids'])
#         print(true_centroids)


if __name__ == '__main__':
    unittest.main()
