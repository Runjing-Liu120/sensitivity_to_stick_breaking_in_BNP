import autograd.numpy as np
import autograd.scipy as sp


class Bspline():
    # Implementing 1D B-splines with Cox-de Boor recursion formula

    def __init__(self, knot_vector):
        # multiplicity of knots not implemented
        assert len(np.unique(knot_vector)) == len(knot_vector)

        # in case the knot vector is not sorted
        self.knot_vector = np.sort(knot_vector)


    def _get_zeroth_basis(self, x):
        # returns the a matrix of size n_basis x len(x)
        # where each row is the nth basis evaluated at x

        # if u_0, ..., u_{n_knots}, are the knots
        # then the nth basis is 1 if x \in [u_{n - 1}, u_n)
        # and 0 otherwise

        zeroth_basis = np.zeros((len(self.knot_vector), len(x)))
        pos_basis_indx = np.searchsorted(self.knot_vector, x)
        pos_basis_indx[pos_basis_indx == len(self.knot_vector)] = \
                            len(self.knot_vector) - 1
        zeroth_basis[pos_basis_indx, np.arange(len(x))] = 1.0

        return zeroth_basis[1:, ]

    def get_pth_order_basis(self, x, p):
        # Recursive Cox - de Boor function to get basis vectors
        # x should be a vector
        # we return a matrix of size n_basis x len(x)

        assert len(np.shape(x)) == 1 # x should be a vector

        if p == 0:
            return self._get_zeroth_basis(x)
        else:
            basis_p_minus_1 = self.get_pth_order_basis(x, p - 1)


        first_term_numerator = x[np.newaxis, :] - self.knot_vector[:-p][:, np.newaxis]
        first_term_denominator = self.knot_vector[p:] - self.knot_vector[:-p]
        # print(first_term_denominator)

        second_term_numerator = self.knot_vector[(p + 1):][:, np.newaxis] - x[np.newaxis, :]
        second_term_denominator = (self.knot_vector[(p + 1):] -
                                   self.knot_vector[1:-p])

        first_term = first_term_numerator / first_term_denominator[:, np.newaxis]

        second_term = second_term_numerator / second_term_denominator[:, np.newaxis]

        return  (first_term[:-1] * basis_p_minus_1[:-1] +
                 second_term * basis_p_minus_1[1:])

class BsplineRegression():
    def __init__(self, fun, x_reg, knot_vector, order):
        # fun is the funtion we wish to approximate
        # x_reg are the x_values of the function at which we wish to
        # evaluate the basis vectors, and use these to compute the coefficients
        # knot_vector and order define the b-spline basis

        # to evalute the function call eval_fun(x)
        # to evalute the b-spline approximation call eval_interp_fun(x)

        self.bsplines = Bspline(knot_vector)
        self.order = order

        # regressors defined by bsplines
        self.bspline_basis_at_x_reg = \
            self.bsplines.get_pth_order_basis(x_reg, self.order)

        # function values at x_reg
        self.fun_at_x_reg = fun(x_reg)

        # get pseudo inverse
        self._set_pseudo_inv()

        # get coefficients
        self.bspline_coeffs = np.dot(self.pseudo_inv, self.fun_at_x_reg)

        self.fun = fun

    def _set_pseudo_inv(self):
        ev = np.linalg.eigvals(np.dot(
                        self.bspline_basis_at_x_reg,
                        self.bspline_basis_at_x_reg.T))
        print('condition number: ', ev[0] / ev[-1])
        assert ev[-1] > 1e-12 # any better checks for this?

        self.pseudo_inv = np.linalg.pinv(self.bspline_basis_at_x_reg.T)

    def eval_fun(self, x):
        return self.fun(x)

    def eval_interp_fun(self, x):
        return np.dot(self.bspline_coeffs, \
                        self.bsplines.get_pth_order_basis(x, self.order))
