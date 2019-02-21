import numpy as np

import scipy as sp
from scipy import spatial
import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import paragami

from copy import deepcopy

def load_data(demean=True):
    iris = datasets.load_iris(return_X_y= True)
    iris_features = iris[0]
    demean = True
    if demean:
        iris_features -= np.mean(iris_features, axis = 0)[None, :]

    iris_species = iris[1]
    return iris_features, iris_species


def plot_clusters(x, y, cluster_labels, colors, fig, centroids = None, cov = None):
    if np.all(cov != None):
        assert len(np.unique(cluster_labels)) == np.shape(cov)[0]
    if np.all(centroids != None):
        assert len(np.unique(cluster_labels)) == np.shape(centroids)[1]

    unique_cluster_labels = np.unique(cluster_labels)
    n_clusters = len(unique_cluster_labels)

    # this would be so much easier if
    # python lists supported logical indexing ...
    cluster_labels_color = [colors[k] for n in range(len(x)) \
                            for k in range(n_clusters) \
                            if cluster_labels[n] == unique_cluster_labels[k]]

    # plot datapoints
    fig.scatter(x, y, c=cluster_labels_color, marker = '.')

    if np.all(centroids != None):
        for k in range(n_clusters):
            fig.scatter(centroids[0, k], centroids[1, k], marker = '+', color = 'black')

    if np.all(cov != None):
        for k in range(n_clusters):
            eig, v = np.linalg.eig(cov[k, :, :])
            ell = Ellipse(xy=(centroids[0, k], centroids[1, k]),
                  width=np.sqrt(eig[0]) * 6, height=np.sqrt(eig[1]) * 6,
                  angle=np.rad2deg(np.arctan(v[1, 0] / v[0, 0])))
            ell.set_facecolor('none')
            ell.set_edgecolor(colors[k])
            fig.add_artist(ell)


def transform_params_to_pc_space(pca_fit, centroids, cov):
    # PCA fit should be the output of
    # pca_fit = PCA()
    # pca_fit.fit(iris_features)

    # centroids is dim x k_approx
    # infos is k_approx x dim x dim

    assert pca_fit.components_.shape[1] == centroids.shape[0]

    centroids_pc = pca_fit.transform(centroids.T)

    cov_pc = np.zeros(cov.shape)
    for k in range(cov.shape[0]):
        cov_pc[k, :, :] = np.dot(np.dot(pca_fit.components_, cov[k]), pca_fit.components_.T)

    # cov_pc = np.einsum('di, kij, ej -> kde', pca_fit.components_, cov, pca_fit.components_)

    return centroids_pc.T, cov_pc


def get_plot_data(iris_features):
    # Define some things that will be useful for plotting.

    # define colors that will be used for plotting later
    # colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    # colors += colors

    pca_fit = PCA()
    pca_fit.fit(iris_features)
    pc_features = pca_fit.transform(iris_features)

    cmap = cm.get_cmap(name='gist_rainbow')
    colors1 = [cmap(k * 50) for k in range(12)]
    colors2 = [cmap(k * 25) for k in range(12)]
    return pca_fit, pc_features, colors1, colors2

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
