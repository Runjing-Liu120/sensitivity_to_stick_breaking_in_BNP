import autograd
import autograd.numpy as np
import autograd.scipy as sp

from scipy import optimize

from sklearn.cluster import KMeans

from copy import deepcopy

import paragami

##########################
# Optimization functions
##########################

def run_bfgs(get_loss, init_vb_free_params,
                    get_loss_grad =  None,
                    maxiter = 10, gtol = 1e-8):

    """
    Runs BFGS to find the optimal variational parameters

    Parameters
    ----------
    get_loss : Callable function
        A callable function that takes in the variational free parameters
        and returns the negative ELBO.
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
        optimization.
    get_loss_grad : Callable function (optional)
        A callable function that takes in the variational free parameters
        and returns the gradient of get_loss.
    maxiter : int
        Maximum number of iterations to run bfgs.
    gtol : float
        The tolerance used to check that the gradient is approximately
            zero at the optimum.

    Returns
    -------
    bfgs_vb_free_params : vec
        Vector of optimal variational free parameters.
    bfgs_output :
        The OptimizeResult class from returned by scipy.optimize.minimize.

    """

    if get_loss_grad is None:
        get_loss_grad = autograd.grad(get_loss)

    # optimize
    bfgs_output = optimize.minimize(
            get_loss,
            x0=init_vb_free_params,
            jac=get_loss_grad,
            method='BFGS',
            options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})

    bfgs_vb_free_params = bfgs_output.x

    return bfgs_vb_free_params, bfgs_output

def precondition_and_optimize(get_loss, init_vb_free_params,
                                maxiter = 10, gtol = 1e-8):
    """
    Finds a preconditioner at init_vb_free_params, and then
    runs trust Newton conjugate gradient to find the optimal
    variational parameters.

    Parameters
    ----------
    get_loss : Callable function
        A callable function that takes in the variational free parameters
        and returns the negative ELBO.
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
        optimization.
    get_loss_grad : Callable function (optional)
        A callable function that takes in the variational free parameters
        and returns the gradient of get_loss.
    maxiter : int
        Maximum number of iterations to run Newton
    gtol : float
        The tolerance used to check that the gradient is approximately
            zero at the optimum.

    Returns
    -------
    bfgs_vb_free_params : vec
        Vector of optimal variational free parameters.
    bfgs_output : class OptimizeResult from scipy.Optimize

    """

    # get preconditioned function
    print('computing preconditioner ')
    precond_fun = paragami.PreconditionedFunction(get_loss)
    _ = precond_fun.set_preconditioner_with_hessian(x = init_vb_free_params,
                                                        ev_min=1e-4)

    # optimize
    print('running newton steps')
    trust_ncg_output = optimize.minimize(
                            method='trust-ncg',
                            x0=precond_fun.precondition(init_vb_free_params),
                            fun=precond_fun,
                            jac=autograd.grad(precond_fun),
                            hessp=autograd.hessian_vector_product(precond_fun),
                            options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})

    # Uncondition
    trust_ncg_vb_free_pars = precond_fun.unprecondition(trust_ncg_output.x)

    return trust_ncg_vb_free_pars, trust_ncg_output

def optimize_full(get_loss, init_vb_free_params,
                    bfgs_max_iter = 50, netwon_max_iter = 50,
                    max_precondition_iter = 10,
                    gtol=1e-8, ftol=1e-8, xtol=1e-8):
    """
    Finds the optimal variational free parameters of using a combination of
    BFGS and Newton trust region conjugate gradient.

    Runs a few BFGS steps, and computes a preconditioner at the BFGS optimum.
    After preconditioning, we run Newton trust region conjugate gradient.
    If the tolerance is not satisfied after Newton steps, we compute another
    preconditioner and repeat.

    Parameters
    ----------
    get_loss : Callable function
        A callable function that takes in the variational free parameters
        and returns the negative ELBO.
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
    bfgs_max_iter : int
        Maximum number of iterations to run initial BFGS.
    newton_max_iter : int
        Maximum number of iterations to run Newton steps.
    max_precondition_iter : int
        Maximum number of times to recompute preconditioner.
    ftol : float
        The tolerance used to check that the difference in function value
        is approximately zero at the last step.
    xtol : float
        The tolerance used to check that the difference in x values in the L
        infinity norm is approximately zero at the last step.
    gtol : float
        The tolerance used to check that the gradient is approximately
            zero at the optimum.

    Returns
    -------
    vec
        A vector of optimal variational free parameters.

    """

    # compute the gradient
    get_loss_grad = autograd.grad(get_loss)

    # run a few steps of bfgs
    print('running bfgs ... ')
    bfgs_vb_free_params, bfgs_ouput = run_bfgs(get_loss,
                                init_vb_free_params,
                                maxiter = bfgs_max_iter, gtol = gtol)
    x = bfgs_vb_free_params
    f_val = get_loss(x)

    if bfgs_ouput.success:
        print('bfgs converged. Done. ')
        return x

    else:
        # if bfgs did not converge, we precondition and run newton trust region
        for i in range(max_precondition_iter):
            print('\n running preconditioned newton; iter = ', i)
            new_x, ncg_output = precondition_and_optimize(get_loss, x,\
                                        maxiter = netwon_max_iter, gtol = gtol)

            # Check convergence.
            new_f_val = get_loss(new_x)
            grad_val = get_loss_grad(new_x)

            x_diff = np.sum(np.abs(new_x - x))
            f_diff = np.abs(new_f_val - f_val)
            grad_l1 = np.sum(np.abs(grad_val))
            x_conv = x_diff < xtol
            f_conv = f_diff < ftol
            grad_conv = grad_l1 < gtol

            x = new_x
            f_val = new_f_val

            converged = x_conv or f_conv or grad_conv or ncg_output.success

            print('Iter {}: x_diff = {}, f_diff = {}, grad_l1 = {}'.format(
                i, x_diff, f_diff, grad_l1))

            if converged:
                print('done. ')
                break

        return new_x
