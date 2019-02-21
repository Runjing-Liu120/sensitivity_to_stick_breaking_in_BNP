#!/usr/bin/env python3

import sys
sys.path.insert(0, '../')

import autograd.numpy as np
import autograd.scipy as sp

import cluster_quantities_lib
import unittest

import numpy.testing as testing

class TestClusteringSamples(unittest.TestCase):
    def test_sampling_clusters_from_uniform(self):
        # check the sampling of _get_clusters_from_ez_and_unif_samples
        np.random.seed(24524)

        # cluster belonging probabilities
        n_obs = 5
        n_clusters = 3
        e_z = np.random.random((n_obs, n_clusters))
        e_z = e_z / np.sum(e_z, axis = 1, keepdims = True)
        e_z_cumsum = e_z.cumsum(1)

        # draw uniform samples
        n_samples = 100000
        unif_samples = np.random.random((n_obs, n_samples))

        # get cluster belongings from uniform samples
        z_ind_samples = \
            cluster_quantities_lib._get_clusters_from_ez_and_unif_samples(\
                                        e_z_cumsum, unif_samples)

        # sample
        e_z_sampled = np.zeros(e_z.shape)
        for i in range(n_clusters):
            e_z_sampled[:, i] = (z_ind_samples == i).mean(axis = 1)

        tol = 3 * np.sqrt(e_z * (1 - e_z) / n_samples)

        assert np.all(np.abs(e_z_sampled - e_z) < tol), \
                    'diff is {}'.format(np.max(np.abs(e_z_sampled - e_z)))

    def test_get_e_num_clusters_from_ez(self):
        # check that get_e_num_clusters_from_ez, which computes
        # the expected number of clusters via MC matches the
        # analytic expectation

        np.random.seed(54654)

        n_obs = 5
        n_clusters = 3
        e_z = np.random.random((n_obs, n_clusters))
        e_z = e_z / np.sum(e_z, axis = 1, keepdims = True)

        e_num_clusters_sampled, var_num_clusters_sampled = \
            cluster_quantities_lib.get_e_num_large_clusters_from_ez(e_z,
                                                    n_samples = 10000,
                                                    unif_samples = None,
                                                    threshold = 0)

        e_num_clusters_analytic = \
            cluster_quantities_lib.get_e_num_clusters_from_ez(e_z)

        testing.assert_allclose(e_num_clusters_analytic,
                                e_num_clusters_sampled,
                                atol = np.sqrt(var_num_clusters_sampled) * 3,
                                rtol = 0)
if __name__ == '__main__':
    unittest.main()
