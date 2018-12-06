import autograd
import autograd.numpy as np
import autograd.scipy as sp

from scipy import optimize

from sklearn.cluster import KMeans

from copy import deepcopy

import modeling_lib as modeling_lib
import functional_sensitivity_lib as fun_sens_lib

import paragami

##########################
# Set up vb parameters
##########################

def get_vb_params_paragami_object(dim, k_approx):
    """
    Returns a paragami patterned dictionary
    that stores the variational parameters.

    Parameters
    ----------
    dim : int
        Dimension of the datapoints.
    k_approx : int
        Number of components in the model.

    Returns
    -------
    vb_params_dict : dictionary
        A dictionary that contains the variational parameters.

    vb_params_paragami : paragami patterned dictionary
        A paragami patterned dictionary that contains the variational parameters.

    """

    vb_params_paragami = paragami.PatternDict()

    # cluster centroids
    vb_params_paragami['centroids'] = \
        paragami.NumericArrayPattern(shape=(dim, k_approx))

    # BNP sticks
    # variational distribution for each stick is logitnormal
    vb_params_paragami['stick_propn_mean'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,))
    vb_params_paragami['stick_propn_info'] = \
        paragami.NumericArrayPattern(shape = (k_approx - 1,), lb = 1e-4)

    # cluster covariances
    vb_params_paragami['gamma'] = \
        paragami.pattern_containers.PatternArray(shape = (k_approx, ), \
                    base_pattern = paragami.PSDSymmetricMatrixPattern(size=dim))

    vb_params_dict = vb_params_paragami.random()

    return vb_params_dict, vb_params_paragami


##########################
# Set up prior parameters
##########################
def get_default_prior_params(dim):
    """
    Returns a paragami patterned dictionary
    that stores the prior parameters.

    Default prior parameters are those set for the experiments in
    "Evaluating Sensitivity to the Stick Breaking Prior in
    Bayesian Nonparametrics"
    https://arxiv.org/abs/1810.06587

    Parameters
    ----------
    dim : int
        Dimension of the datapoints.

    Returns
    -------
    prior_params_dict : dictionary
        A dictionary that contains the prior parameters.

    prior_params_paragami : paragami Patterned Dictionary
        A paragami patterned dictionary that contains the prior parameters.

    """

    prior_params_dict = dict()
    prior_params_paragami = paragami.PatternDict()

    # DP prior parameter
    prior_params_dict['alpha'] = np.array([4.0])
    prior_params_paragami['alpha'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the centroids
    prior_params_dict['prior_centroid_mean'] = np.array([0.0])
    prior_params_paragami['prior_centroid_mean'] = \
        paragami.NumericArrayPattern(shape=(1, ))

    prior_params_dict['prior_centroid_info'] = np.array([0.1])
    prior_params_paragami['prior_centroid_info'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    # prior on the variance
    prior_params_dict['prior_gamma_df'] = np.array([8.0])
    prior_params_paragami['prior_gamma_df'] = \
        paragami.NumericArrayPattern(shape=(1, ), lb = 0.0)

    prior_params_dict['prior_gamma_inv_scale'] = 0.62 * np.eye(dim)
    prior_params_paragami['prior_gamma_inv_scale'] = \
        paragami.PSDSymmetricMatrixPattern(size=dim)

    return prior_params_dict, prior_params_paragami

##########################
# Expected prior term
##########################
def get_e_log_prior(stick_propn_mean, stick_propn_info, centroids, gamma,
                        prior_params_dict,
                        gh_loc, gh_weights):
    # get expected prior term

    # dp prior
    alpha = prior_params_dict['alpha']
    dp_prior = \
        modeling_lib.get_e_logitnorm_dp_prior(stick_propn_mean, stick_propn_info,
                                            alpha, gh_loc, gh_weights)

    # wishart prior
    df = prior_params_dict['prior_gamma_df']
    V_inv = prior_params_dict['prior_gamma_inv_scale']
    e_gamma_prior = modeling_lib.get_e_log_wishart_prior(gamma, df, V_inv)

    # centroid prior
    prior_mean = prior_params_dict['prior_centroid_mean']
    prior_info = prior_params_dict['prior_centroid_info']
    e_centroid_prior = \
        modeling_lib.get_e_centroid_prior(centroids, prior_mean, prior_info)

    return np.squeeze(e_gamma_prior + e_centroid_prior + dp_prior)

##########################
# Entropy
##########################
def get_entropy(stick_propn_mean, stick_propn_info, e_z, gh_loc, gh_weights,
                    use_logitnormal_sticks = True):
    # get entropy term

    z_entropy = modeling_lib.multinom_entropy(e_z)
    stick_entropy = \
        modeling_lib.get_stick_breaking_entropy(stick_propn_mean, stick_propn_info,
                                gh_loc, gh_weights)

    return z_entropy + stick_entropy

##########################
# Likelihood term
##########################
def get_loglik_obs_by_nk(y, centroids, gamma):
    # returns a n x k matrix whose nkth entry is
    # the likelihood for the nth observation
    # belonging to the kth cluster

    dim = np.shape(y)[1]

    assert np.shape(y)[1] == np.shape(centroids)[0]
    assert np.shape(gamma)[0] == np.shape(centroids)[1]
    assert np.shape(gamma)[1] == np.shape(centroids)[0]

    data2_term = np.einsum('ni, kij, nj -> nk', y, gamma, y)
    cross_term = np.einsum('ni, kij, jk -> nk', y, gamma, centroids)
    centroid2_term = np.einsum('ik, kij, jk -> k', centroids, gamma, centroids)

    squared_term = data2_term - 2 * cross_term + \
                    np.expand_dims(centroid2_term, axis = 0)

    return - 0.5 * squared_term + 0.5 * np.linalg.slogdet(gamma)[1][None, :]

##########################
# Optimization over e_z
##########################

def get_z_nat_params(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                        gh_loc, gh_weights,
                        use_bnp_prior = True):

    # get likelihood term
    loglik_obs_by_nk = get_loglik_obs_by_nk(y, centroids, gamma)

    # get weight term
    if use_bnp_prior:
        e_log_cluster_probs = \
            modeling_lib.get_e_log_cluster_probabilities(
                            stick_propn_mean, stick_propn_info,
                            gh_loc, gh_weights)
    else:
        e_log_cluster_probs = 0.

    z_nat_param = loglik_obs_by_nk + e_log_cluster_probs

    return z_nat_param, loglik_obs_by_nk

def get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                    gh_loc, gh_weights,
                    use_bnp_prior = True):

    z_nat_param, loglik_obs_by_nk= \
        get_z_nat_params(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                                    gh_loc, gh_weights,
                                    use_bnp_prior)

    log_const = sp.misc.logsumexp(z_nat_param, axis=1)
    e_z = np.exp(z_nat_param - log_const[:, None])

    return e_z, loglik_obs_by_nk

def get_optimal_z_from_vb_params_dict(y, vb_params_dict, gh_loc, gh_weights,
                                        use_bnp_prior = True):

    """
    Returns the optimal cluster belonging probabilities, given the
    variational parameters.

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_dict : dictionary
        Dictionary of variational parameters.
    gh_loc : vector
        Locations for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    gh_weights : vector
        Weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    e_z : ndarray
        The optimal cluster belongings as a function of the variational
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k

    """

    # get global vb parameters
    stick_propn_mean = vb_params_dict['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_propn_info']
    centroids = vb_params_dict['centroids']
    gamma = vb_params_dict['gamma']

    # compute optimal e_z from vb global parameters
    e_z, _ = get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                        gh_loc, gh_weights,
                        use_bnp_prior = True)

    return e_z


def get_kl(y, vb_params_dict, prior_params_dict,
                    gh_loc, gh_weights,
                    e_z = None,
                    data_weights = None,
                    use_bnp_prior = True):

    """
    Computes the negative ELBO using the data y, at the current variational
    parameters and at the current prior parameters

    Parameters
    ----------
    y : ndarray
        The array of datapoints, one observation per row.
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
        parameters, stored in an array whose (n, k)th entry is the probability
        of the nth datapoint belonging to cluster k.
        If ``None``, we set the optimal z.
    data_weights : ndarray of shape (number of observations) x 1 (optional)
        Weights for each datapoint in y.
    use_bnp_prior : boolean
        Whether or not to use a prior on the cluster mixture weights.
        If True, a DP prior is used.

    Returns
    -------
    kl : float
        The negative elbo.
    """
    # get vb parameters
    stick_propn_mean = vb_params_dict['stick_propn_mean']
    stick_propn_info = vb_params_dict['stick_propn_info']
    centroids = vb_params_dict['centroids']
    gamma = vb_params_dict['gamma']

    # get optimal cluster belongings
    e_z_opt, loglik_obs_by_nk = \
            get_optimal_z(y, stick_propn_mean, stick_propn_info, centroids, gamma,
                            gh_loc, gh_weights)
    if e_z is None:
        e_z = e_z_opt


    # weight data if necessary, and get likelihood of y
    if data_weights is not None:
        assert np.shape(data_weights)[0] == n_obs, \
                    'data weights need to be n_obs by 1'
        assert np.shape(data_weights)[1] == 1, \
                    'data weights need to be n_obs by 1'
        e_loglik_obs = np.sum(data_weights * e_z * loglik_obs_by_nk)
    else:
        e_loglik_obs = np.sum(e_z * loglik_obs_by_nk)

    # likelihood of z
    if use_bnp_prior:
        e_loglik_ind = modeling_lib.loglik_ind(stick_propn_mean, stick_propn_info, e_z,
                            gh_loc, gh_weights)
    else:
        e_loglik_ind = 0.

    e_loglik = e_loglik_ind + e_loglik_obs

    if not np.isfinite(e_loglik):
        print('gamma', vb_params_dict['gamma'].get())
        print('det gamma', np.linalg.slogdet(
            vb_params_dict['gamma'])[1])
        print('cluster weights', np.sum(e_z, axis = 0))

    assert(np.isfinite(e_loglik))

    # entropy term
    entropy = np.squeeze(get_entropy(stick_propn_mean, stick_propn_info, e_z,
                                        gh_loc, gh_weights))
    assert(np.isfinite(entropy))

    # prior term
    e_log_prior = get_e_log_prior(stick_propn_mean, stick_propn_info, centroids, gamma,
                            prior_params_dict,
                            gh_loc, gh_weights)

    assert(np.isfinite(e_log_prior))

    elbo = e_log_prior + entropy + e_loglik

    return -1 * elbo

##########################
# Optimization functions
##########################
def cluster_and_get_k_means_inits(y, vb_params_paragami,
                                n_kmeans_init = 1,
                                z_init_eps=0.05):
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

        if len(indx) == 1:
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



def run_bfgs(get_vb_free_params_loss, init_vb_free_params,
                    get_vb_free_params_loss_grad =  None,
                    maxiter = 10, gtol = 1e-8):

    """
    Runs BFGS to find the optimal variational parameters

    Parameters
    ----------
    get_vb_free_params_loss : Callable function
        A callable function that takes in the variational free parameters
        and returns the negative ELBO.
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
        optimization.
    get_vb_free_params_loss_grad : Callable function (optional)
        A callable function that takes in the variational free parameters
        and returns the gradient of get_vb_free_params_loss.
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

    if get_vb_free_params_loss_grad is None:
        get_vb_free_params_loss_grad = autograd.grad(get_vb_free_params_loss)

    # optimize
    bfgs_output = optimize.minimize(
            get_vb_free_params_loss,
            x0=init_vb_free_params,
            jac=get_vb_free_params_loss_grad,
            method='BFGS',
            options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})

    bfgs_vb_free_params = bfgs_output.x

    return bfgs_vb_free_params, bfgs_output

def precondition_and_optimize(get_vb_free_params_loss, init_vb_free_params,
                                maxiter = 10, gtol = 1e-8):
    """
    Finds a preconditioner at init_vb_free_params, and then
    runs trust Newton conjugate gradient to find the optimal
    variational parameters.

    Parameters
    ----------
    get_vb_free_params_loss : Callable function
        A callable function that takes in the variational free parameters
        and returns the negative ELBO.
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
        optimization.
    get_vb_free_params_loss_grad : Callable function (optional)
        A callable function that takes in the variational free parameters
        and returns the gradient of get_vb_free_params_loss.
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
    precond_fun = paragami.PreconditionedFunction(get_vb_free_params_loss)
    _ = precond_fun.set_preconditioner_with_hessian(x = init_vb_free_params,
                                                        ev_min=1e-4)

    # optimize
    trust_ncg_output = optimize.minimize(
                            method='trust-ncg',
                            x0=precond_fun.precondition(init_vb_free_params),
                            fun=precond_fun,
                            jac=autograd.grad(precond_fun),
                            hess=autograd.hessian(precond_fun),
                            options={'maxiter': maxiter, 'disp': True, 'gtol': gtol})

    # Uncondition
    trust_ncg_vb_free_pars = precond_fun.unprecondition(trust_ncg_output.x)

    return trust_ncg_vb_free_pars, trust_ncg_output

def optimize_full(y, vb_params_paragami, prior_params_dict,
                    init_vb_free_params, gh_loc, gh_weights,
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
    y : ndarray
        The array of datapoints, one observation per row.
    vb_params_paragami : paragami Patterned Dictionary
        a paragami patterned dictionary that contains the variational parameters
    prior_params_dict : dictionary
        a dictionary that contains the prior parameters
    init_vb_free_params : vector
        Vector of the free variational parameters at which we initialize the
    gh_loc : vector
        locations for gauss-hermite quadrature. We need this compute the
        expected prior terms
    gh_weights : vector
        weights for gauss-hermite quadrature. We need this compute the
        expected prior terms.
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


    # Get loss as a function of the  vb_params_dict
    get_vb_params_loss = paragami.Functor(original_fun=get_kl, argnums=1)
    get_vb_params_loss.cache_args(y, None, prior_params_dict,
                                    gh_loc, gh_weights)

    # Get loss as a function vb_free_params
    get_vb_free_params_loss = paragami.FlattenedFunction(
                                                original_fun=get_vb_params_loss,
                                                patterns=vb_params_paragami,
                                                free=True)
    # get gradient
    get_vb_free_params_loss_grad = autograd.grad(get_vb_free_params_loss)
    get_vb_free_params_loss_hess = autograd.hessian(get_vb_free_params_loss)

    # run a few steps of bfgs
    print('running bfgs ... ')
    bfgs_vb_free_params, bfgs_ouput = run_bfgs(get_vb_free_params_loss,
                                init_vb_free_params,
                                get_vb_free_params_loss_grad,
                                maxiter = bfgs_max_iter, gtol = gtol)
    x = bfgs_vb_free_params
    f_val = get_vb_free_params_loss(x)

    if bfgs_ouput.success:
        print('bfgs converged. Done. ')
        return x

    else:
        # if bfgs did not converge, we precondition and run newton trust region
        for i in range(max_precondition_iter):
            print('\n running preconditioned newton; iter = ', i)
            new_x, ncg_output = precondition_and_optimize(get_vb_free_params_loss, x,\
                                        maxiter = netwon_max_iter, gtol = gtol)

            # Check convergence.
            new_f_val = get_vb_free_params_loss(new_x)
            grad_val = get_vb_free_params_loss_grad(new_x)

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


########################
# Posterior quantities of interest
#######################

def get_e_num_pred_clusters_from_vb_free_params(vb_params_paragami,
                                                    vb_params_free,
                                                    n_obs,
                                                    threshold = 0,
                                                    n_samples = 100000):
    # get posterior predicted number of clusters

    _, vb_params_dict = \
        get_moments_from_vb_free_params(vb_params_paragami, vb_params_free)

    mu = vb_params_dict['stick_propn_mean']
    sigma = 1 / np.sqrt(vb_params_dict['stick_propn_info'])

    return modeling_lib.get_e_number_clusters_from_logit_sticks(mu, sigma,
                                                        n_obs,
                                                        threshold = threshold,
                                                        n_samples = n_samples)


# Get the expected posterior number of distinct clusters.
def get_e_num_clusters_from_free_par(y, vb_params_paragami, vb_params_free,
                                        gh_loc, gh_weights,
                                        threshold = 0,
                                        n_samples = 100000):

    _, vb_params_dict = \
        get_moments_from_vb_free_params(vb_params_paragami, vb_params_free)

    e_z  = get_optimal_z_from_vb_params_dict(y, vb_params_dict, gh_loc, gh_weights,
                                            use_bnp_prior = True)

    return modeling_lib.get_e_num_large_clusters_from_ez(e_z,
                                        threshold = threshold,
                                        n_samples = 100000)
