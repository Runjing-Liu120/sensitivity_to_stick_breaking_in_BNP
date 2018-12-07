#!/usr/bin/env python3

import sys
sys.path.insert(0, '../libraries')

import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

import modeling_lib

from numpy.polynomial.hermite import hermgauss

import unittest

import numpy.testing as testing

class TestModelingLib(unittest.TestCase):
    def test_logitnorm_entropy(self):
        np.random.seed(456456)

        gh_deg = 8
        gh_loc, gh_weights = hermgauss(gh_deg)

        # set means and variances
        mu = np.random.randn(10)
        info = np.exp(np.random.randn(10))

        # compute entropy
        logitnormal_entropy = \
            modeling_lib.get_stick_breaking_entropy(mu, info, gh_loc, gh_weights)

        # sample from logitnormal
        n_samples = 100000
        normal_samples = np.random.randn(n_samples, len(mu))
        logitnorm_samples = \
            osp.special.expit(normal_samples * 1/ np.sqrt(info) + mu)

        # get logintormal pdf
        log_pdf_samples = \
            osp.stats.norm.logpdf(osp.special.logit(logitnorm_samples), \
                                    loc = mu, scale = 1 / np.sqrt(info)) - \
            np.log(logitnorm_samples) - np.log(1 - logitnorm_samples)

        # get sampled entropy
        sampled_entropy = np.sum(-log_pdf_samples, axis = 1)

        # check diff in mean
        mean_sampled_entropy = np.mean(sampled_entropy)
        # diff = np.abs(logitnormal_entropy - mean_sampled_entropy)
        # assert diff < 0.01, 'diff = {}'.format(diff)
        testing.assert_allclose(mean_sampled_entropy, logitnormal_entropy,
                                atol = 1e-2, rtol = 0)


if __name__ == '__main__':
    unittest.main()
