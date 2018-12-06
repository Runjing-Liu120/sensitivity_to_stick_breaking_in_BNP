#!/usr/bin/env python3

import sys
sys.path.insert(0, '../')

import autograd.numpy as np
import autograd.scipy as sp

from autograd import elementwise_grad

import interpolation_lib as interp
import unittest

import numpy.testing as testing

# we will try to approximate this function in our tests
fun = lambda x : sp.stats.norm.pdf(x, 0.7, 0.1) - \
                    1.5 * sp.stats.norm.pdf(x, 0.75, 0.1)

# the knots we will use
knot_vector = np.linspace(0.0, 1.0, 1000)

# the regression object
bspline_regression = interp.BsplineRegression(fun, knot_vector, \
                                                knot_vector, 3)

class TestInterpolation(unittest.TestCase):
    def test_spline_derivatives(self):
        np.random.seed(24524)

        # we'll check the gradient at these points
        x = np.random.random(1000)

        # compute derivative with autodiff
        get_bspline_grad_ad = elementwise_grad(bspline_regression.eval_interp_fun)
        bspline_grad_ad = get_bspline_grad_ad(x)

        # compute derivative with finite differences
        h = 1e-10
        bspline_grad_fin_diff = \
            (bspline_regression.eval_interp_fun(x + h) - \
                bspline_regression.eval_interp_fun(x)) / h

        testing.assert_allclose(bspline_grad_ad, bspline_grad_fin_diff,\
                                    atol = 1e-4, rtol = 0)

    def test_fun_interpolation(self):
        # we'll check the interpolation at these points
        x = np.linspace(0.01, 0.99, 100)

        testing.assert_allclose(bspline_regression.eval_fun(x), \
                                bspline_regression.eval_interp_fun(x),\
                                    atol = 1e-4, rtol = 0)

if __name__ == '__main__':
    unittest.main()
