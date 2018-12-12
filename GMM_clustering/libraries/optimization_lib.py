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
def cluster_and_get_k_means_inits(y, vb_params_paragami,
                                n_kmeans_init = 1,
                                z_init_eps=0.05,
                                seed = 1):
    """
    Runs k-means to initialize the variational parameters.

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the variational parameters.
    n_kmeans_init : int
        The number of re-initializations for K-means.
    z_init_eps : float
        The weight given to the clusters a data does not belong to
        after running K-means

    Returns
    -------
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    init_free_par : vector
        Vector of the free variational parameters
    e_z_init : ndarray
        Array encoding cluster belongings as found by kmeans
    """

    # get dictionary of vb parameters
    vb_params_dict = vb_params_paragami.random()

    # set seed
    np.random.seed(seed)

    # data parameters
    k_approx = np.shape(vb_params_dict['centroids'])[1]
    n_obs = np.shape(y)[0]
    dim = np.shape(y)[1]

    # K means init.
    for i in range(n_kmeans_init):
        km = KMeans(n_clusters = k_approx).fit(y)
        enertia = km.inertia_
        if (i == 0):
            enertia_best = enertia
            km_best = deepcopy(km)
        elif (enertia < enertia_best):
            enertia_best = enertia
            km_best = deepcopy(km)

    e_z_init = np.full((n_obs, k_approx), z_init_eps)
    for n in range(len(km_best.labels_)):
        e_z_init[n, km_best.labels_[n]] = 1.0 - z_init_eps
    e_z_init /= np.expand_dims(np.sum(e_z_init, axis = 1), axis = 1)

    vb_params_dict['centroids'] = km_best.cluster_centers_.T

    vb_params_dict['stick_propn_mean'] = np.ones(k_approx - 1)
    vb_params_dict['stick_propn_info'] = np.ones(k_approx - 1)

    # Set inital covariances
    gamma_init = np.zeros((k_approx, dim, dim))
    for k in range(k_approx):
        indx = np.argwhere(km_best.labels_ == k).flatten()

        if len(indx == 1):
            # if there's only one datapoint in the cluster,
            # the covariance is not defined.
            gamma_init[k, :, :] = np.eye(dim)
        else:
            resid_k = y[indx, :] - km_best.cluster_centers_[k, :]
            gamma_init_ = np.linalg.inv(np.cov(resid_k.T) + \
                                    np.eye(dim) * 1e-4)
            # symmetrize ... there might be some numerical issues otherwise
            gamma_init[k, :, :] = 0.5 * (gamma_init_ + gamma_init_.T)

    vb_params_dict['gamma'] = gamma_init
    
    init_free_par = vb_params_paragami.flatten(vb_params_dict, free = True)

    return init_free_par, vb_params_dict, e_z_init



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
